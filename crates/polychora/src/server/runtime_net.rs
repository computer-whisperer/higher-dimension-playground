use super::*;
use crate::shared::spatial::ChunkCoord;

fn world_bootstrap_diag_enabled() -> bool {
    static ENABLED: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    *ENABLED.get_or_init(|| match std::env::var("R4D_WORLD_BOOTSTRAP_DIAG") {
        Ok(value) => {
            let value = value.trim().to_ascii_lowercase();
            !(value.is_empty()
                || value == "0"
                || value == "false"
                || value == "off"
                || value == "no")
        }
        Err(_) => false,
    })
}

fn summarize_region_kind(kind: &crate::shared::region_tree::RegionNodeKind) -> &'static str {
    match kind {
        crate::shared::region_tree::RegionNodeKind::Empty => "Empty",
        crate::shared::region_tree::RegionNodeKind::Uniform(_) => "Uniform",
        crate::shared::region_tree::RegionNodeKind::ChunkArray(_) => "ChunkArray",
        crate::shared::region_tree::RegionNodeKind::Branch(_) => "Branch",
        crate::shared::region_tree::RegionNodeKind::ProceduralRef(_) => "ProceduralRef",
    }
}

fn send_to_client(state: &SharedState, client_id: u64, message: ServerMessage) {
    let sender = {
        let guard = state.lock().expect("server state lock poisoned");
        guard.clients.get(&client_id).cloned()
    };
    if let Some(tx) = sender {
        let _ = tx.send(message);
    }
}

fn prune_stale_clients(state: &SharedState, stale: Vec<u64>, notify_entity_destroyed: bool) {
    if stale.is_empty() {
        return;
    }
    let mut disconnected_entity_ids = Vec::new();
    {
        let mut guard = state.lock().expect("server state lock poisoned");
        for client_id in stale {
            guard.clients.remove(&client_id);
            guard.client_world_interest_bounds.remove(&client_id);
            guard.client_visible_entities.remove(&client_id);
            if let Some(player) = guard.players.remove(&client_id) {
                mark_entity_record_despawned(&mut guard, player.entity_id, None);
                let _ = guard.entity_store.despawn(player.entity_id);
                disconnected_entity_ids.push(player.entity_id);
            }
        }
    }
    if notify_entity_destroyed {
        for entity_id in disconnected_entity_ids {
            broadcast(state, ServerMessage::EntityDestroyed { entity_id });
        }
    }
}

fn broadcast(state: &SharedState, message: ServerMessage) {
    let clients: Vec<_> = {
        let guard = state.lock().expect("server state lock poisoned");
        guard
            .clients
            .iter()
            .map(|(&client_id, tx)| (client_id, tx.clone()))
            .collect()
    };

    let mut stale = Vec::new();
    for (client_id, tx) in clients {
        if tx.send(message.clone()).is_err() {
            stale.push(client_id);
        }
    }

    prune_stale_clients(state, stale, true);
}

fn send_world_subtree_patch_to_client(
    state: &SharedState,
    client_id: u64,
    bounds: Aabb4i,
    allow_empty: bool,
) {
    if !bounds.is_valid() {
        send_to_client(
            state,
            client_id,
            ServerMessage::Error {
                message: format!(
                    "invalid world subtree request bounds {:?}->{:?}",
                    bounds.min, bounds.max
                ),
            },
        );
        return;
    }

    let query_start = Instant::now();
    let subtree = {
        let mut guard = state.lock().expect("server state lock poisoned");
        guard.query_world_subtree(bounds)
    };
    let query_ms = query_start.elapsed().as_secs_f64() * 1000.0;
    if query_ms >= 50.0 {
        eprintln!(
            "[server-world-patch-query-slow] client_id={} bounds={:?}->{:?} elapsed_ms={:.3} allow_empty={}",
            client_id,
            bounds.min,
            bounds.max,
            query_ms,
            allow_empty
        );
    }
    if !allow_empty
        && matches!(
            subtree.kind,
            crate::shared::region_tree::RegionNodeKind::Empty
        )
    {
        if world_bootstrap_diag_enabled() {
            eprintln!(
                "[server-world-bootstrap] skip-empty-patch client_id={} bounds={:?}->{:?} allow_empty={}",
                client_id,
                bounds.min,
                bounds.max,
                allow_empty
            );
        }
        return;
    }
    if world_bootstrap_diag_enabled() {
        eprintln!(
            "[server-world-bootstrap] send-patch client_id={} requested={:?}->{:?} subtree={:?}->{:?} kind={} allow_empty={}",
            client_id,
            bounds.min,
            bounds.max,
            subtree.bounds.min,
            subtree.bounds.max,
            summarize_region_kind(&subtree.kind),
            allow_empty
        );
    }

    send_to_client(
        state,
        client_id,
        ServerMessage::WorldSubtreePatch {
            authoritative_bounds: bounds,
            subtree: (*subtree).clone(),
        },
    );
}

fn intersect_bounds(a: Aabb4i, b: Aabb4i) -> Option<Aabb4i> {
    if !a.intersects(&b) {
        return None;
    }
    let intersection = Aabb4i::new(
        [
            a.min[0].max(b.min[0]),
            a.min[1].max(b.min[1]),
            a.min[2].max(b.min[2]),
            a.min[3].max(b.min[3]),
        ],
        [
            a.max[0].min(b.max[0]),
            a.max[1].min(b.max[1]),
            a.max[2].min(b.max[2]),
            a.max[3].min(b.max[3]),
        ],
    );
    if intersection.is_valid() {
        Some(intersection)
    } else {
        None
    }
}

fn subtract_bounds(outer: Aabb4i, inner: Aabb4i) -> Vec<Aabb4i> {
    let Some(inner) = intersect_bounds(outer, inner) else {
        return vec![outer];
    };
    if inner == outer {
        return Vec::new();
    }

    let mut pieces = Vec::with_capacity(8);
    let mut core = outer;
    let one = ChunkCoord::from_num(1);
    for axis in 0..4 {
        if core.min[axis] < inner.min[axis] {
            let mut piece = core;
            piece.max[axis] = inner.min[axis] - one;
            if piece.is_valid() {
                pieces.push(piece);
            }
            core.min[axis] = inner.min[axis];
        }
        if core.max[axis] > inner.max[axis] {
            let mut piece = core;
            piece.min[axis] = inner.max[axis] + one;
            if piece.is_valid() {
                pieces.push(piece);
            }
            core.max[axis] = inner.max[axis];
        }
    }
    pieces
}

fn split_bounds_for_streaming(
    bounds: Aabb4i,
    max_chunk_cells_per_patch: usize,
    max_patches: usize,
) -> Vec<Aabb4i> {
    if !bounds.is_valid() {
        return Vec::new();
    }
    if max_patches <= 1 {
        return vec![bounds];
    }

    let mut out = Vec::<Aabb4i>::new();
    let mut stack = vec![bounds];
    while let Some(current) = stack.pop() {
        let Some(cell_count) = current.chunk_cell_count() else {
            out.push(current);
            continue;
        };
        if cell_count <= max_chunk_cells_per_patch || out.len() + stack.len() + 1 >= max_patches {
            out.push(current);
            continue;
        }

        let one = ChunkCoord::from_num(1);
        let two = ChunkCoord::from_num(2);
        let extents = [
            current.max[0] - current.min[0] + one,
            current.max[1] - current.min[1] + one,
            current.max[2] - current.min[2] + one,
            current.max[3] - current.min[3] + one,
        ];
        let mut split_axis = 0usize;
        for axis in 1..4 {
            if extents[axis] > extents[split_axis] {
                split_axis = axis;
            }
        }
        if extents[split_axis] <= one {
            out.push(current);
            continue;
        }

        let mid = current.min[split_axis] + (extents[split_axis] / two) - one;
        let mut left = current;
        let mut right = current;
        left.max[split_axis] = mid;
        right.min[split_axis] = mid + one;
        if left.is_valid() {
            stack.push(left);
        }
        if right.is_valid() {
            stack.push(right);
        }
    }
    out
}

fn broadcast_world_dirty_bounds_updates(state: &SharedState, dirty_bounds: &[Aabb4i]) -> usize {
    if dirty_bounds.is_empty() {
        return 0;
    }
    let mut unique = dirty_bounds.to_vec();
    unique.sort_unstable_by_key(|bounds| {
        (
            bounds.min[0],
            bounds.min[1],
            bounds.min[2],
            bounds.min[3],
            bounds.max[0],
            bounds.max[1],
            bounds.max[2],
            bounds.max[3],
        )
    });
    unique.dedup();

    let recipients_by_client = {
        let guard = state.lock().expect("server state lock poisoned");
        guard
            .client_world_interest_bounds
            .iter()
            .map(|(&client_id, &interest)| (client_id, interest))
            .collect::<Vec<_>>()
    };

    let mut recipients_by_clip = HashMap::<Aabb4i, Vec<u64>>::new();
    for bounds in unique {
        for (client_id, interest) in &recipients_by_client {
            let Some(clipped) = intersect_bounds(bounds, *interest) else {
                continue;
            };
            recipients_by_clip
                .entry(clipped)
                .or_default()
                .push(*client_id);
        }
    }
    if recipients_by_clip.is_empty() {
        return 0;
    }

    let mut sent = 0usize;
    for (clip_bounds, mut client_ids) in recipients_by_clip {
        client_ids.sort_unstable();
        client_ids.dedup();
        let subtree = {
            let mut guard = state.lock().expect("server state lock poisoned");
            guard.query_world_subtree(clip_bounds)
        };
        for client_id in client_ids {
            send_to_client(
                state,
                client_id,
                ServerMessage::WorldSubtreePatch {
                    authoritative_bounds: clip_bounds,
                    subtree: (*subtree).clone(),
                },
            );
            sent = sent.saturating_add(1);
        }
    }
    sent
}

fn send_empty_world_subtree_patch_to_client(state: &SharedState, client_id: u64, bounds: Aabb4i) {
    if !bounds.is_valid() {
        return;
    }
    send_to_client(
        state,
        client_id,
        ServerMessage::WorldSubtreePatch {
            authoritative_bounds: bounds,
            subtree: crate::shared::region_tree::RegionTreeCore {
                bounds,
                kind: crate::shared::region_tree::RegionNodeKind::Empty,
                generator_version_hash: 0,
            },
        },
    );
}

fn zero_dense_chunk_blocks() -> Vec<voxel::BlockData> {
    vec![voxel::BlockData::AIR; voxel::CHUNK_VOLUME]
}

fn dense_blocks_from_resolved_payload(
    resolved: &crate::shared::chunk_payload::ResolvedChunkPayload,
) -> Vec<voxel::BlockData> {
    let Ok(palette_indices) = resolved.payload.dense_materials() else {
        return zero_dense_chunk_blocks();
    };
    if palette_indices.len() != voxel::CHUNK_VOLUME {
        return zero_dense_chunk_blocks();
    }
    palette_indices
        .iter()
        .map(|&idx| {
            resolved
                .block_palette
                .get(idx as usize)
                .cloned()
                .unwrap_or(voxel::BlockData::AIR)
        })
        .collect()
}

fn dense_blocks_from_region_core_chunk(
    core: &crate::shared::region_tree::RegionTreeCore,
    chunk_key: [i32; 4],
) -> Vec<voxel::BlockData> {
    use crate::shared::region_tree::chunk_key_i32;
    let ck = chunk_key_i32(chunk_key[0], chunk_key[1], chunk_key[2], chunk_key[3]);
    let chunk_bounds = Aabb4i::new(ck, ck);
    let chunks = crate::shared::region_tree::collect_non_empty_chunks_from_core_in_bounds(
        core,
        chunk_bounds,
    );
    for (key, resolved) in chunks {
        if key == ck {
            return dense_blocks_from_resolved_payload(&resolved);
        }
    }
    zero_dense_chunk_blocks()
}

fn handle_world_chunk_sample_request(
    state: &SharedState,
    client_id: u64,
    request_id: u64,
    chunk: [i32; 4],
) {
    let chunk_bounds = Aabb4i::from_i32(chunk, chunk);
    if !chunk_bounds.is_valid() {
        send_to_client(
            state,
            client_id,
            ServerMessage::Error {
                message: format!("invalid world chunk sample request chunk={chunk:?}"),
            },
        );
        return;
    }

    let dense_blocks = {
        let mut guard = state.lock().expect("server state lock poisoned");
        let subtree = guard.query_world_subtree(chunk_bounds);
        dense_blocks_from_region_core_chunk(subtree.as_ref(), chunk)
    };

    send_to_client(
        state,
        client_id,
        ServerMessage::WorldChunkSampleResponse {
            request_id,
            chunk,
            dense_blocks,
        },
    );
}

fn handle_world_interest_update(state: &SharedState, client_id: u64, bounds: Aabb4i) {
    if !bounds.is_valid() {
        send_to_client(
            state,
            client_id,
            ServerMessage::Error {
                message: format!(
                    "invalid world interest bounds {:?}->{:?}",
                    bounds.min, bounds.max
                ),
            },
        );
        return;
    }

    let previous = {
        let mut guard = state.lock().expect("server state lock poisoned");
        guard.set_client_world_interest_bounds(client_id, bounds)
    };
    if world_bootstrap_diag_enabled() {
        eprintln!(
            "[server-world-bootstrap] interest-update client_id={} bounds={:?}->{:?} previous={:?}",
            client_id,
            bounds.min,
            bounds.max,
            previous.as_ref().map(|prev| (prev.min, prev.max))
        );
    }
    if previous == Some(bounds) {
        if world_bootstrap_diag_enabled() {
            eprintln!(
                "[server-world-bootstrap] interest-noop client_id={} bounds={:?}->{:?}",
                client_id, bounds.min, bounds.max
            );
        }
        return;
    }

    let added_bounds = match previous {
        Some(prev) => subtract_bounds(bounds, prev),
        None => vec![bounds],
    };
    let removed_bounds = match previous {
        Some(prev) => subtract_bounds(prev, bounds),
        None => Vec::new(),
    };
    let removed_count = removed_bounds.len();
    let added_count = added_bounds.len();

    for remove_bounds in removed_bounds {
        send_empty_world_subtree_patch_to_client(state, client_id, remove_bounds);
    }
    let allow_empty_add = previous.is_none();
    if world_bootstrap_diag_enabled() {
        eprintln!(
            "[server-world-bootstrap] interest-delta client_id={} adds={} removals={} allow_empty_add={}",
            client_id,
            added_count,
            removed_count,
            allow_empty_add
        );
    }
    for add_bounds in added_bounds {
        const WORLD_PATCH_MAX_CHUNK_CELLS: usize = 2_000_000;
        const WORLD_PATCH_MAX_SPLITS: usize = 256;
        let add_slices = split_bounds_for_streaming(
            add_bounds,
            WORLD_PATCH_MAX_CHUNK_CELLS,
            WORLD_PATCH_MAX_SPLITS,
        );
        if world_bootstrap_diag_enabled() && add_slices.len() > 1 {
            eprintln!(
                "[server-world-bootstrap] interest-split client_id={} source={:?}->{:?} slices={} max_cells_per_patch={}",
                client_id,
                add_bounds.min,
                add_bounds.max,
                add_slices.len(),
                WORLD_PATCH_MAX_CHUNK_CELLS
            );
        }
        for slice_bounds in add_slices {
            send_world_subtree_patch_to_client(state, client_id, slice_bounds, allow_empty_add);
        }
    }
}

fn apply_authoritative_voxel_edit(
    state: &mut ServerState,
    position: [i32; 4],
    block: BlockData,
) -> Option<crate::shared::region_tree::ChunkKey> {
    state.apply_world_voxel_edit(position, block)
}

fn flush_world_dirty_updates(state: &SharedState) -> usize {
    let dirty_bounds = {
        let mut guard = state.lock().expect("server state lock poisoned");
        guard.world_take_dirty_bounds()
    };
    broadcast_world_dirty_bounds_updates(state, &dirty_bounds)
}

pub(super) fn apply_creeper_explosion(
    state: &mut ServerState,
    source_entity_id: u64,
    center: [f32; 4],
    radius_voxels: i32,
) -> (usize, QueuedExplosionEvent) {
    let radius = radius_voxels.max(1);
    let radius_sq = radius * radius;
    let center_voxel = [
        center[0].floor() as i32,
        center[1].floor() as i32,
        center[2].floor() as i32,
        center[3].floor() as i32,
    ];
    let blast_centers = [
        center_voxel,
        [
            center_voxel[0],
            center_voxel[1] - 1,
            center_voxel[2],
            center_voxel[3],
        ],
    ];

    let mut changed_chunks = HashSet::new();
    for blast_center in blast_centers {
        for dx in -radius..=radius {
            for dy in -radius..=radius {
                for dz in -radius..=radius {
                    for dw in -radius..=radius {
                        let dist_sq = dx * dx + dy * dy + dz * dz + dw * dw;
                        if dist_sq > radius_sq {
                            continue;
                        }
                        let pos = [
                            blast_center[0] + dx,
                            blast_center[1] + dy,
                            blast_center[2] + dz,
                            blast_center[3] + dw,
                        ];
                        if let Some(chunk_pos) =
                            apply_authoritative_voxel_edit(state, pos, BlockData::AIR)
                        {
                            changed_chunks.insert(chunk_pos);
                        }
                    }
                }
            }
        }
    }

    (
        changed_chunks.len(),
        QueuedExplosionEvent {
            position: center,
            radius: radius as f32,
            source_entity_id: Some(source_entity_id),
        },
    )
}

pub(super) fn apply_explosion_impulse(
    state: &mut ServerState,
    source_entity_id: u64,
    center: [f32; 4],
    impulse_radius: f32,
    max_push_distance: f32,
    now_ms: u64,
) -> (bool, Vec<QueuedPlayerMovementModifier>) {
    if !impulse_radius.is_finite()
        || !max_push_distance.is_finite()
        || impulse_radius <= 0.0
        || max_push_distance <= 0.0
    {
        return (false, Vec::new());
    }
    let impulse_radius_sq = impulse_radius * impulse_radius;
    let mut pending_impulses = Vec::new();
    let mut queued_player_modifiers = Vec::new();
    for record in state.entity_records.values() {
        if record.lifecycle != EntityLifecycle::Live || record.entity_id == source_entity_id {
            continue;
        }
        let Some(snapshot) = state.entity_store.snapshot(record.entity_id) else {
            continue;
        };
        let pos = snapshot.entity.pose.position;
        let orient = snapshot.entity.pose.orientation;
        let offset = [
            pos[0] - center[0],
            pos[1] - center[1],
            pos[2] - center[2],
            pos[3] - center[3],
        ];
        let distance_sq = offset[0] * offset[0]
            + offset[1] * offset[1]
            + offset[2] * offset[2]
            + offset[3] * offset[3];
        if !distance_sq.is_finite() || distance_sq > impulse_radius_sq {
            continue;
        }
        let distance = distance_sq.sqrt();
        let falloff = (1.0 - distance / impulse_radius).clamp(0.0, 1.0);
        if falloff <= 0.0 {
            continue;
        }
        let push_distance = max_push_distance * falloff * falloff;
        if push_distance <= 1e-3 {
            continue;
        }
        let outward = if distance > 1e-4 {
            [
                offset[0] / distance,
                offset[1] / distance,
                offset[2] / distance,
                offset[3] / distance,
            ]
        } else {
            normalize4_or_default(orient, [1.0, 0.0, 0.0, 0.0])
        };
        let next_position = [
            pos[0] + outward[0] * push_distance,
            pos[1] + outward[1] * push_distance,
            pos[2] + outward[2] * push_distance,
            pos[3] + outward[3] * push_distance,
        ];
        let next_orientation = normalize4_or_default(outward, orient);
        pending_impulses.push((
            record.entity_id,
            next_position,
            next_orientation,
            record.persistent,
        ));
    }
    for (&client_id, player) in &state.players {
        if player.entity_id == source_entity_id {
            continue;
        }
        let Some(snapshot) = state.entity_store.snapshot(player.entity_id) else {
            continue;
        };
        let pos = snapshot.entity.pose.position;
        let orient = snapshot.entity.pose.orientation;
        let offset = [
            pos[0] - center[0],
            pos[1] - center[1],
            pos[2] - center[2],
            pos[3] - center[3],
        ];
        let distance_sq = offset[0] * offset[0]
            + offset[1] * offset[1]
            + offset[2] * offset[2]
            + offset[3] * offset[3];
        if !distance_sq.is_finite() || distance_sq > impulse_radius_sq {
            continue;
        }
        let distance = distance_sq.sqrt();
        let falloff = (1.0 - distance / impulse_radius).clamp(0.0, 1.0);
        if falloff <= 0.0 {
            continue;
        }
        let push_distance = max_push_distance * falloff * falloff;
        if push_distance <= 1e-3 {
            continue;
        }
        let outward = if distance > 1e-4 {
            [
                offset[0] / distance,
                offset[1] / distance,
                offset[2] / distance,
                offset[3] / distance,
            ]
        } else {
            normalize4_or_default(orient, [1.0, 0.0, 0.0, 0.0])
        };
        let delta_position = [
            outward[0] * push_distance,
            outward[1] * push_distance,
            outward[2] * push_distance,
            outward[3] * push_distance,
        ];
        let delta_velocity_y = (outward[1] * push_distance * 6.0).clamp(-18.0, 18.0);
        queued_player_modifiers.push(QueuedPlayerMovementModifier {
            client_id,
            delta_position,
            delta_velocity_y,
            source_entity_id: Some(source_entity_id),
        });
    }
    if queued_player_modifiers.is_empty() && !state.players.is_empty() {
        eprintln!(
            "[impulse-debug][server] explosion produced no player modifiers source_entity={} center={:?} players_connected={}",
            source_entity_id,
            center,
            state.players.len()
        );
    }

    let mut persistent_motion = false;
    for (entity_id, next_position, next_orientation, persistent) in pending_impulses {
        if !state
            .entity_store
            .set_motion_state(entity_id, next_position, next_orientation, now_ms)
        {
            continue;
        }
        if persistent {
            persistent_motion = true;
        }
    }
    (persistent_motion, queued_player_modifiers)
}

pub(super) fn start_broadcast_thread(
    state: SharedState,
    tick_hz: f32,
    entity_sim_hz: f32,
    entity_interest_radius_chunks: i32,
    save_interval_secs: u64,
    start: Instant,
    shutdown: Arc<AtomicBool>,
    mut wasm_manager: Option<crate::shared::wasm::WasmPluginManager>,
) {
    let interval = Duration::from_secs_f64(1.0 / tick_hz.max(0.1) as f64);
    let save_interval = Duration::from_secs(save_interval_secs.max(1));
    let entity_sim_step_ms = (1000.0 / entity_sim_hz.max(0.1) as f64).round().max(1.0) as u64;
    let entity_interest_radius_sq = {
        let radius = entity_interest_radius_chunks.max(0) as i64;
        radius * radius
    };
    let mut next_entity_sim_ms = 0u64;
    let mut last_save_tick = Instant::now();
    thread::spawn(move || {
        while !shutdown.load(Ordering::Relaxed) {
            thread::sleep(interval);
            if shutdown.load(Ordering::Relaxed) {
                break;
            }
            let tick_cpu_start = Instant::now();
            let now = monotonic_ms(start);
            let (entity_batches, explosion_events, player_movement_modifiers, sim_timings) = {
                let mut guard = state.lock().expect("server state lock poisoned");
                let _ = guard.evict_far_mob_nav_cache_chunks(
                    MOB_NAV_CACHE_KEEP_RADIUS_CHUNKS,
                    MOB_NAV_CACHE_EVICT_BUDGET_PER_TICK,
                );
                let (explosion_events, player_movement_modifiers, sim_timings) = tick_entity_simulation_window(
                    &mut guard,
                    &mut wasm_manager,
                    now,
                    &mut next_entity_sim_ms,
                    entity_sim_step_ms,
                );
                let entity_batches =
                    build_entity_replication_batches(&mut guard, entity_interest_radius_sq);
                (entity_batches, explosion_events, player_movement_modifiers, sim_timings)
            };
            let spawned_count: usize = entity_batches.iter().map(|batch| batch.spawned.len()).sum();
            let transform_count: usize = entity_batches
                .iter()
                .map(|batch| batch.transforms.len())
                .sum();
            let world_chunk_update_count = flush_world_dirty_updates(&state);
            let explosion_count = explosion_events.len();
            let player_modifier_count = player_movement_modifiers.len();
            let did_broadcast = entity_batches.iter().any(|batch| {
                !batch.spawned.is_empty()
                    || !batch.despawned.is_empty()
                    || !batch.transforms.is_empty()
            }) || world_chunk_update_count > 0
                || explosion_count > 0
                || player_modifier_count > 0;

            for explosion in explosion_events {
                broadcast(
                    &state,
                    ServerMessage::Explosion {
                        position: explosion.position,
                        radius: explosion.radius,
                        source_entity_id: explosion.source_entity_id,
                    },
                );
            }
            for modifier in player_movement_modifiers {
                let delta = modifier.delta_position;
                let delta_len = (delta[0] * delta[0]
                    + delta[1] * delta[1]
                    + delta[2] * delta[2]
                    + delta[3] * delta[3])
                    .sqrt();
                eprintln!(
                    "[impulse-debug][server] send_modifier client_id={} source_entity={:?} delta={:?} delta_len={:.3} delta_velocity_y={:.3}",
                    modifier.client_id,
                    modifier.source_entity_id,
                    modifier.delta_position,
                    delta_len,
                    modifier.delta_velocity_y
                );
                send_to_client(
                    &state,
                    modifier.client_id,
                    ServerMessage::PlayerMovementModifier {
                        delta_position: modifier.delta_position,
                        delta_velocity_y: modifier.delta_velocity_y,
                        source_entity_id: modifier.source_entity_id,
                    },
                );
            }

            for batch in entity_batches {
                for entity_id in batch.despawned {
                    send_to_client(
                        &state,
                        batch.client_id,
                        ServerMessage::EntityDestroyed { entity_id },
                    );
                }
                for entity in batch.spawned {
                    send_to_client(
                        &state,
                        batch.client_id,
                        ServerMessage::EntitySpawned { entity },
                    );
                }
                if !batch.transforms.is_empty() {
                    send_to_client(
                        &state,
                        batch.client_id,
                        ServerMessage::EntityTransforms {
                            server_time_ms: now,
                            entities: batch.transforms,
                        },
                    );
                }
            }
            if did_broadcast {
                record_server_cpu_sample(
                    &state,
                    None,
                    Some((
                        tick_cpu_start.elapsed(),
                        spawned_count.saturating_add(explosion_count),
                        transform_count
                            .saturating_add(world_chunk_update_count)
                            .saturating_add(player_modifier_count),
                    )),
                    Some(&sim_timings),
                );
            }

            if last_save_tick.elapsed() >= save_interval {
                let save_result = {
                    let mut guard = state.lock().expect("server state lock poisoned");
                    guard.persist_world_if_dirty(crate::save_v4::now_unix_ms())
                };
                match save_result {
                    Ok(Some(result)) => {
                        eprintln!(
                            "persisted v4 world save generation={} block_regions={} entity_regions={}",
                            result.generation, result.saved_block_regions, result.saved_entity_regions
                        );
                    }
                    Ok(None) => {}
                    Err(error) => {
                        eprintln!("failed to persist v4 world save: {}", error);
                    }
                }
                last_save_tick = Instant::now();
            }
        }
    });
}

pub(super) fn remove_client(state: &SharedState, client_id: u64) {
    let removed_entity_id = {
        let mut guard = state.lock().expect("server state lock poisoned");
        let _ = guard.clients.remove(&client_id);
        guard.client_world_interest_bounds.remove(&client_id);
        guard.client_visible_entities.remove(&client_id);
        match guard.players.remove(&client_id) {
            Some(player) => {
                mark_entity_record_despawned(&mut guard, player.entity_id, None);
                let _ = guard.entity_store.despawn(player.entity_id);
                Some(player.entity_id)
            }
            None => None,
        }
    };
    if let Some(entity_id) = removed_entity_id {
        broadcast(state, ServerMessage::EntityDestroyed { entity_id });
    }
}

fn install_or_update_player(
    state: &SharedState,
    client_id: u64,
    name: Option<String>,
    position: Option<[f32; 4]>,
    look: Option<[f32; 4]>,
    start: Instant,
) -> (EntitySnapshot, bool) {
    let now = monotonic_ms(start);
    let mut guard = state.lock().expect("server state lock poisoned");
    let default_orientation = [
        0.0,
        0.0,
        std::f32::consts::FRAC_1_SQRT_2,
        std::f32::consts::FRAC_1_SQRT_2,
    ];
    let mut spawned_now = false;

    let make_player_entity = |pos: [f32; 4], orient: [f32; 4]| -> Entity {
        Entity {
            namespace: ENTITY_PLAYER_AVATAR.0,
            entity_type: ENTITY_PLAYER_AVATAR.1,
            pose: EntityPose {
                position: pos,
                orientation: orient,
                velocity: [0.0; 4],
                scale: 1.0,
            },
            data: Vec::new(),
        }
    };

    let entity_id = match guard.players.entry(client_id) {
        Entry::Occupied(entry) => entry.get().entity_id,
        Entry::Vacant(entry) => {
            spawned_now = true;
            let entity_id = client_id;
            entry.insert(PlayerState { entity_id });
            let entity = make_player_entity(
                position.unwrap_or([0.0, 0.0, 0.0, 0.0]),
                look.unwrap_or(default_orientation),
            );
            guard.entity_store.spawn(entity_id, entity, now);
            entity_id
        }
    };

    if guard.entity_store.snapshot(entity_id).is_none() {
        spawned_now = true;
        let entity = make_player_entity(
            position.unwrap_or([0.0, 0.0, 0.0, 0.0]),
            look.unwrap_or(default_orientation),
        );
        guard.entity_store.spawn(entity_id, entity, now);
    }

    let current = guard
        .entity_store
        .snapshot(entity_id)
        .expect("player entity should exist in store");
    let target_position = position.unwrap_or(current.entity.pose.position);
    let target_orientation = look.unwrap_or(current.entity.pose.orientation);
    let _ =
        guard
            .entity_store
            .set_motion_state(entity_id, target_position, target_orientation, now);

    let player_name = name
        .map(|n| sanitize_player_name(&n, client_id))
        .or_else(|| {
            guard
                .entity_records
                .get(&entity_id)
                .and_then(|record| record.display_name.clone())
        })
        .unwrap_or_else(|| format!("player-{client_id}"));

    let mut snapshot = guard
        .entity_store
        .snapshot(entity_id)
        .expect("updated player entity should exist in store");
    snapshot.owner_client_id = Some(client_id);
    snapshot.display_name = Some(player_name.clone());
    upsert_entity_record(
        &mut guard,
        entity_id,
        EntityCategory::Player,
        Some(client_id),
        Some(player_name),
        false,
        now,
    );
    (snapshot, spawned_now)
}

fn spawn_entity_from_request(
    state: &SharedState,
    namespace: u32,
    entity_type: u32,
    position: [f32; 4],
    orientation: [f32; 4],
    scale: f32,
    start: Instant,
) -> Result<EntitySnapshot, String> {
    if (namespace, entity_type) == ENTITY_PLAYER_AVATAR {
        return Err("player entities are server-managed and cannot be spawned".to_string());
    }
    let registry = {
        let guard = state.lock().expect("server state lock poisoned");
        guard.content_registry.clone()
    };
    let Some(entry) = registry.entity_lookup(namespace, entity_type) else {
        return Err(format!(
            "entity type ({namespace}, {entity_type}) is not registered"
        ));
    };
    if !entry.is_spawnable() {
        return Err(format!(
            "entity type '{}' is not spawnable",
            entry.canonical_name
        ));
    }

    let spawn_position = [
        if position[0].is_finite() {
            position[0]
        } else {
            0.0
        },
        if position[1].is_finite() {
            position[1]
        } else {
            0.0
        },
        if position[2].is_finite() {
            position[2]
        } else {
            0.0
        },
        if position[3].is_finite() {
            position[3]
        } else {
            0.0
        },
    ];
    let spawn_orientation = if orientation.iter().all(|axis| axis.is_finite()) {
        normalize4_or_default(orientation, [0.0, 0.0, 1.0, 0.0])
    } else {
        [0.0, 0.0, 1.0, 0.0]
    };
    let spawn_scale = if scale.is_finite() {
        scale.clamp(0.10, 8.0)
    } else {
        0.5
    };

    let entity_data = Entity {
        namespace,
        entity_type,
        pose: EntityPose {
            position: spawn_position,
            orientation: spawn_orientation,
            velocity: [0.0; 4],
            scale: spawn_scale,
        },
        data: Vec::new(),
    };

    let snapshot = match (entry.category, entry.sim_config.as_ref().map(|c| c.mode)) {
        (EntityCategory::Mob, Some(polychora_plugin_api::entity::SimulationMode::PhysicsDriven)) => {
            let config = entry.sim_config.as_ref().unwrap();
            spawn_mob_entity(
                state,
                entity_data,
                namespace,
                entity_type,
                config,
                None,
                true,
                None,
                None,
                start,
            )
        }
        (EntityCategory::Accent, _) => {
            spawn_entity(state, entity_data, None, true, None, start)
        }
        (category, sim_mode) => {
            return Err(format!(
                "spawn spec for '{}' is invalid (category={:?}, sim_mode={:?})",
                entry.canonical_name, category, sim_mode
            ));
        }
    };
    Ok(snapshot)
}

fn handle_console_spawn_command(
    state: &SharedState,
    client_id: u64,
    args: &[&str],
    start: Instant,
) -> Result<(), String> {
    let registry = {
        let guard = state.lock().expect("server state lock poisoned");
        guard.content_registry.clone()
    };
    if args.is_empty() {
        return Err(spawn_usage_string(&registry));
    }
    let Some(entry) = entity_type_entry_for_token(args[0], &registry) else {
        let available = registry.spawnable_entity_names().join(", ");
        return Err(format!(
            "unknown spawn kind '{}'; available kinds: {available}",
            args[0]
        ));
    };

    let (default_position, default_orientation) = {
        let guard = state.lock().expect("server state lock poisoned");
        default_spawn_pose_for_client(&guard, client_id)
    };
    let parse_usage = || spawn_usage_string(&registry);
    let position = match args.len() {
        1 => default_position,
        5 => {
            let Some(position) = parse_spawn_vec4(&args[1..5]) else {
                return Err(parse_usage());
            };
            position
        }
        _ => return Err(parse_usage()),
    };

    let _ = spawn_entity_from_request(
        state,
        entry.namespace,
        entry.entity_type,
        position,
        default_orientation,
        entry.default_scale,
        start,
    )?;
    Ok(())
}

fn run_server_console_command(
    state: &SharedState,
    client_id: u64,
    command: &str,
    start: Instant,
) -> Result<(), String> {
    let mut command = command.trim();
    if command.is_empty() {
        return Ok(());
    }
    if let Some(stripped) = command.strip_prefix('/') {
        command = stripped;
    }
    let mut parts = command.split_whitespace();
    let Some(command_name) = parts.next() else {
        return Ok(());
    };
    let args: Vec<&str> = parts.collect();

    if command_name.eq_ignore_ascii_case("spawn") {
        return handle_console_spawn_command(state, client_id, &args, start);
    }
    Err(format!(
        "unknown server command '{}'; supported: /spawn",
        command_name
    ))
}

pub(super) fn handle_message(
    state: &SharedState,
    client_id: u64,
    message: ClientMessage,
    tick_hz: f32,
    start: Instant,
) {
    let message_cpu_start = Instant::now();
    match message {
        ClientMessage::Hello { name } => {
            let (_snapshot, _spawned_now) =
                install_or_update_player(state, client_id, Some(name), None, None, start);
            send_to_client(
                state,
                client_id,
                ServerMessage::Welcome {
                    client_id,
                    server_time_ms: monotonic_ms(start),
                    tick_hz,
                    world: {
                        let guard = state.lock().expect("server state lock poisoned");
                        WorldSummary {
                            non_empty_chunks: guard.world_non_empty_chunk_count(),
                            bounds: guard.world_bounds(),
                        }
                    },
                },
            );
        }
        ClientMessage::UpdatePlayer { position, look } => {
            let safe_position = if position.iter().all(|axis| axis.is_finite()) {
                position
            } else {
                [0.0, 0.0, 0.0, 0.0]
            };
            let safe_look = normalize4_or_default(
                look,
                [
                    0.0,
                    0.0,
                    std::f32::consts::FRAC_1_SQRT_2,
                    std::f32::consts::FRAC_1_SQRT_2,
                ],
            );
            let (_snapshot, _spawned_now) = install_or_update_player(
                state,
                client_id,
                None,
                Some(safe_position),
                Some(safe_look),
                start,
            );
        }
        ClientMessage::SetVoxel { position, block } => {
            let _changed_chunk = {
                let mut guard = state.lock().expect("server state lock poisoned");
                apply_authoritative_voxel_edit(&mut guard, position, block)
            };
        }
        ClientMessage::SpawnEntity {
            entity_type_namespace,
            entity_type,
            position,
            orientation,
            scale,
        } => {
            if let Err(message) = spawn_entity_from_request(
                state,
                entity_type_namespace,
                entity_type,
                position,
                orientation,
                scale,
                start,
            ) {
                send_to_client(state, client_id, ServerMessage::Error { message });
            }
        }
        ClientMessage::ConsoleCommand { command } => {
            if let Err(message) = run_server_console_command(state, client_id, &command, start) {
                send_to_client(state, client_id, ServerMessage::Error { message });
            }
        }
        ClientMessage::WorldInterestUpdate { bounds } => {
            handle_world_interest_update(state, client_id, bounds);
        }
        ClientMessage::WorldChunkSampleRequest { request_id, chunk } => {
            handle_world_chunk_sample_request(state, client_id, request_id, chunk);
        }
        ClientMessage::Ping { nonce } => {
            send_to_client(state, client_id, ServerMessage::Pong { nonce });
        }
    }
    record_server_cpu_sample(state, Some(message_cpu_start.elapsed()), None, None);
}

pub(super) fn spawn_client_thread(
    stream: TcpStream,
    state: SharedState,
    tick_hz: f32,
    start: Instant,
) {
    let peer_label = stream
        .peer_addr()
        .map(|a| a.to_string())
        .unwrap_or_else(|_| "<unknown>".to_string());
    let (tx, rx) = mpsc::channel::<ServerMessage>();

    let client_id = {
        let mut guard = state.lock().expect("server state lock poisoned");
        let id = allocate_server_object_id(&mut guard);
        guard.clients.insert(id, tx.clone());
        id
    };

    let writer_stream = match stream.try_clone() {
        Ok(s) => s,
        Err(error) => {
            eprintln!("failed to clone stream for {}: {}", peer_label, error);
            remove_client(&state, client_id);
            return;
        }
    };

    thread::spawn(move || {
        let mut writer = BufWriter::new(writer_stream);
        while let Ok(message) = rx.recv() {
            let encoded = match postcard::to_stdvec(&message) {
                Ok(encoded) => encoded,
                Err(error) => {
                    match &message {
                        ServerMessage::WorldSubtreePatch {
                            authoritative_bounds,
                            subtree,
                        } => {
                            eprintln!(
                                "failed to encode world subtree patch for client {} authoritative={:?}->{:?} subtree={:?}->{:?}: {}",
                                client_id,
                                authoritative_bounds.min,
                                authoritative_bounds.max,
                                subtree.bounds.min,
                                subtree.bounds.max,
                                error
                            );
                        }
                        _ => {
                            eprintln!(
                                "failed to encode server message for client {}: {}",
                                client_id, error
                            );
                        }
                    }
                    continue;
                }
            };
            let len = (encoded.len() as u32).to_le_bytes();
            if writer.write_all(&len).is_err() {
                break;
            }
            if writer.write_all(&encoded).is_err() {
                break;
            }
            if writer.flush().is_err() {
                break;
            }
        }
    });

    thread::spawn(move || {
        eprintln!("client {} connected from {}", client_id, peer_label);
        let mut reader = stream;
        let mut len_buf = [0u8; 4];

        loop {
            match reader.read_exact(&mut len_buf) {
                Ok(()) => {}
                Err(error) if error.kind() == io::ErrorKind::UnexpectedEof => {
                    break;
                }
                Err(error) => {
                    eprintln!("read error for client {}: {}", client_id, error);
                    break;
                }
            }

            let len = u32::from_le_bytes(len_buf) as usize;
            if len > 100_000_000 {
                eprintln!(
                    "client {} sent oversized message ({} bytes)",
                    client_id, len
                );
                break;
            }

            let mut msg_buf = vec![0u8; len];
            match reader.read_exact(&mut msg_buf) {
                Ok(()) => {}
                Err(error) => {
                    eprintln!("read error for client {}: {}", client_id, error);
                    break;
                }
            }

            let parsed = postcard::from_bytes::<ClientMessage>(&msg_buf);
            match parsed {
                Ok(message) => {
                    handle_message(&state, client_id, message, tick_hz, start);
                }
                Err(error) => {
                    send_to_client(
                        &state,
                        client_id,
                        ServerMessage::Error {
                            message: format!("invalid message: {error}"),
                        },
                    );
                }
            }
        }

        remove_client(&state, client_id);
        eprintln!("client {} disconnected", client_id);
    });
}

fn spawn_entity(
    state: &SharedState,
    entity: Entity,
    display_name: Option<String>,
    persistent: bool,
    persisted_entity_id: Option<EntityId>,
    start: Instant,
) -> EntitySnapshot {
    let mut guard = state.lock().expect("server state lock poisoned");
    let allocated_id = allocate_or_reserve_server_object_id(&mut guard, persisted_entity_id);
    let now_ms = monotonic_ms(start);
    guard.entity_store.spawn(allocated_id, entity, now_ms);
    upsert_entity_record(
        &mut guard,
        allocated_id,
        EntityCategory::Accent,
        None,
        display_name,
        persistent,
        now_ms,
    );
    guard
        .entity_store
        .snapshot(allocated_id)
        .expect("spawned entity should exist in store")
}

fn spawn_mob_entity(
    state: &SharedState,
    entity: Entity,
    entity_ns: u32,
    entity_type: u32,
    config: &polychora_plugin_api::entity::EntitySimConfig,
    display_name: Option<String>,
    persistent: bool,
    persisted_mob: Option<PersistedMobEntry>,
    persisted_entity_id: Option<EntityId>,
    start: Instant,
) -> EntitySnapshot {
    let mut guard = state.lock().expect("server state lock poisoned");
    let allocated_id = allocate_or_reserve_server_object_id(&mut guard, persisted_entity_id);
    let now_ms = monotonic_ms(start);
    guard
        .entity_store
        .spawn(allocated_id, entity, now_ms);
    let phase_offset = persisted_mob
        .as_ref()
        .map(|mob| mob.phase_offset)
        .unwrap_or_else(|| ((allocated_id as f32) * 0.73).rem_euclid(std::f32::consts::TAU));
    let move_speed = persisted_mob
        .as_ref()
        .map(|mob| mob.move_speed)
        .unwrap_or(config.move_speed)
        .max(0.1);
    let preferred_distance = persisted_mob
        .as_ref()
        .map(|mob| mob.preferred_distance)
        .unwrap_or(config.preferred_distance)
        .max(0.1);
    let tangent_weight = persisted_mob
        .as_ref()
        .map(|mob| mob.tangent_weight)
        .unwrap_or(config.tangent_weight)
        .clamp(0.0, 2.0);
    let initial_phase_tick_seed = (allocated_id as u32).wrapping_mul(7477);
    let blink_min_ms = config.ability_params.as_ref().map(|a| a.blink_min_interval_ms).unwrap_or(0);
    let blink_max_ms = config.ability_params.as_ref().map(|a| a.blink_max_interval_ms).unwrap_or(0);
    let next_phase_ms =
        phase_spider_next_phase_deadline(now_ms, phase_offset, initial_phase_tick_seed, blink_min_ms, blink_max_ms);
    guard.mobs.insert(
        allocated_id,
        MobState {
            entity_id: allocated_id,
            entity_ns,
            entity_type,
            phase_offset,
            move_speed,
            preferred_distance,
            tangent_weight,
            next_phase_ms,
            navigation: MobNavigationState::default(),
        },
    );
    upsert_entity_record(
        &mut guard,
        allocated_id,
        EntityCategory::Mob,
        None,
        display_name,
        persistent,
        now_ms,
    );
    guard
        .entity_store
        .snapshot(allocated_id)
        .expect("spawned mob entity should exist in store")
}
