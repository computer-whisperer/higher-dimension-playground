use super::*;

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

fn force_sync_streamed_clients_for_changed_chunks(
    state: &SharedState,
    changed_chunks: &[ChunkPos],
    source_client_id: Option<u64>,
    near_chunk_radius: i32,
    mid_chunk_radius: i32,
    far_chunk_radius: i32,
) {
    // TODO(region-tree-streaming): Replace this temporary no-op with true per-client
    // region-tree delta planning and patch emission.
    let _ = (
        state,
        changed_chunks,
        source_client_id,
        near_chunk_radius,
        mid_chunk_radius,
        far_chunk_radius,
    );
}

fn sync_streamed_chunks_for_client(
    state: &SharedState,
    client_id: u64,
    center_chunk: [i32; 4],
    near_chunk_radius: i32,
    mid_chunk_radius: i32,
    far_chunk_radius: i32,
    force: bool,
) {
    // TODO(region-tree-streaming): Replace this with a true client/server demand stream.
    let _ = (
        state,
        client_id,
        center_chunk,
        near_chunk_radius,
        mid_chunk_radius,
        far_chunk_radius,
        force,
    );
}

fn send_world_subtree_patch_to_client(state: &SharedState, client_id: u64, bounds: Aabb4i) {
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

    let subtree = {
        let guard = state.lock().expect("server state lock poisoned");
        guard
            .world
            .query_region_core(QueryVolume { bounds }, QueryDetail::Exact)
    };

    send_to_client(
        state,
        client_id,
        ServerMessage::WorldSubtreePatch {
            subtree: (*subtree).clone(),
        },
    );
}

fn apply_authoritative_voxel_edit(
    state: &mut ServerState,
    position: [i32; 4],
    material: VoxelType,
) -> Option<ChunkPos> {
    state.world.apply_voxel_edit(position, material)
}

pub(super) fn apply_creeper_explosion(
    state: &mut ServerState,
    source_entity_id: u64,
    center: [f32; 4],
    radius_voxels: i32,
) -> (Vec<QueuedWorldChunkUpdate>, QueuedExplosionEvent) {
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
                            apply_authoritative_voxel_edit(state, pos, VoxelType::AIR)
                        {
                            changed_chunks.insert(chunk_pos);
                        }
                    }
                }
            }
        }
    }

    let mut world_updates = Vec::new();
    if !changed_chunks.is_empty() {
        let mut changed_chunks: Vec<ChunkPos> = changed_chunks.into_iter().collect();
        changed_chunks.sort_unstable_by_key(|chunk| (chunk.w, chunk.z, chunk.y, chunk.x));
        world_updates.push(QueuedWorldChunkUpdate { changed_chunks });
    }

    (
        world_updates,
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
        let offset = [
            snapshot.position[0] - center[0],
            snapshot.position[1] - center[1],
            snapshot.position[2] - center[2],
            snapshot.position[3] - center[3],
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
            normalize4_or_default(snapshot.orientation, [1.0, 0.0, 0.0, 0.0])
        };
        let next_position = [
            snapshot.position[0] + outward[0] * push_distance,
            snapshot.position[1] + outward[1] * push_distance,
            snapshot.position[2] + outward[2] * push_distance,
            snapshot.position[3] + outward[3] * push_distance,
        ];
        let next_orientation = normalize4_or_default(outward, snapshot.orientation);
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
        let offset = [
            snapshot.position[0] - center[0],
            snapshot.position[1] - center[1],
            snapshot.position[2] - center[2],
            snapshot.position[3] - center[3],
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
            normalize4_or_default(snapshot.orientation, [1.0, 0.0, 0.0, 0.0])
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
    near_chunk_radius: i32,
    mid_chunk_radius: i32,
    far_chunk_radius: i32,
    start: Instant,
    shutdown: Arc<AtomicBool>,
) {
    let interval = Duration::from_secs_f64(1.0 / tick_hz.max(0.1) as f64);
    let entity_sim_step_ms = (1000.0 / entity_sim_hz.max(0.1) as f64).round().max(1.0) as u64;
    let entity_interest_radius_sq = {
        let radius = entity_interest_radius_chunks.max(0) as i64;
        radius * radius
    };
    let mut next_entity_sim_ms = 0u64;
    thread::spawn(move || {
        while !shutdown.load(Ordering::Relaxed) {
            thread::sleep(interval);
            if shutdown.load(Ordering::Relaxed) {
                break;
            }
            let tick_cpu_start = Instant::now();
            let now = monotonic_ms(start);
            let (entity_batches, world_chunk_updates, explosion_events, player_movement_modifiers) = {
                let mut guard = state.lock().expect("server state lock poisoned");
                let (world_chunk_updates, explosion_events, player_movement_modifiers) =
                    tick_entity_simulation_window(
                        &mut guard,
                        now,
                        &mut next_entity_sim_ms,
                        entity_sim_step_ms,
                    );
                let entity_batches =
                    build_entity_replication_batches(&mut guard, entity_interest_radius_sq);
                (
                    entity_batches,
                    world_chunk_updates,
                    explosion_events,
                    player_movement_modifiers,
                )
            };
            let spawned_count: usize = entity_batches.iter().map(|batch| batch.spawned.len()).sum();
            let transform_count: usize = entity_batches
                .iter()
                .map(|batch| batch.transforms.len())
                .sum();
            let world_chunk_update_count = world_chunk_updates.len();
            let explosion_count = explosion_events.len();
            let player_modifier_count = player_movement_modifiers.len();
            let did_broadcast = entity_batches.iter().any(|batch| {
                !batch.spawned.is_empty()
                    || !batch.despawned.is_empty()
                    || !batch.transforms.is_empty()
            }) || world_chunk_update_count > 0
                || explosion_count > 0
                || player_modifier_count > 0;

            for update in world_chunk_updates {
                force_sync_streamed_clients_for_changed_chunks(
                    &state,
                    &update.changed_chunks,
                    None,
                    near_chunk_radius,
                    mid_chunk_radius,
                    far_chunk_radius,
                );
            }
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
                );
            }
        }
    });
}

pub(super) fn remove_client(state: &SharedState, client_id: u64) {
    let removed_entity_id = {
        let mut guard = state.lock().expect("server state lock poisoned");
        let _ = guard.clients.remove(&client_id);
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
    let entity_id = match guard.players.entry(client_id) {
        Entry::Occupied(entry) => entry.get().entity_id,
        Entry::Vacant(entry) => {
            spawned_now = true;
            let entity_id = client_id;
            entry.insert(PlayerState { entity_id });
            guard.entity_store.spawn(
                entity_id,
                EntityClass::Player,
                EntityKind::PlayerAvatar,
                position.unwrap_or([0.0, 0.0, 0.0, 0.0]),
                look.unwrap_or(default_orientation),
                1.0,
                0,
                now,
            );
            entity_id
        }
    };

    if guard.entity_store.snapshot(entity_id).is_none() {
        spawned_now = true;
        guard.entity_store.spawn(
            entity_id,
            EntityClass::Player,
            EntityKind::PlayerAvatar,
            position.unwrap_or([0.0, 0.0, 0.0, 0.0]),
            look.unwrap_or(default_orientation),
            1.0,
            0,
            now,
        );
    }

    let current = guard
        .entity_store
        .snapshot(entity_id)
        .expect("player entity should exist in store");
    let target_position = position.unwrap_or(current.position);
    let target_orientation = look.unwrap_or(current.orientation);
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
    snapshot.class = EntityClass::Player;
    snapshot.kind = EntityKind::PlayerAvatar;
    snapshot.scale = 1.0;
    snapshot.material = 0;
    snapshot.owner_client_id = Some(client_id);
    snapshot.display_name = Some(player_name.clone());
    upsert_entity_record(
        &mut guard,
        entity_id,
        EntityClass::Player,
        Some(client_id),
        Some(player_name),
        false,
        now,
    );
    (snapshot, spawned_now)
}

fn spawn_entity_from_request(
    state: &SharedState,
    kind: EntityKind,
    position: [f32; 4],
    orientation: [f32; 4],
    scale: f32,
    material: u8,
    start: Instant,
) -> Result<EntitySnapshot, String> {
    if kind == EntityKind::PlayerAvatar {
        return Err("player entities are server-managed and cannot be spawned".to_string());
    }
    let Some(spec) = spawnable_entity_spec_for_kind(kind) else {
        return Err(format!("entity kind {:?} is not spawnable", kind));
    };

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
    let spawn_material = material.clamp(1, materials::MAX_MATERIAL_ID);

    let entity = match (spec.class, spec.mob_archetype) {
        (EntityClass::Mob, Some(archetype)) => spawn_mob_entity(
            state,
            kind,
            archetype,
            spawn_position,
            spawn_orientation,
            spawn_scale,
            spawn_material,
            None,
            true,
            None,
            None,
            start,
        ),
        (EntityClass::Accent, None) => spawn_entity(
            state,
            kind,
            spawn_position,
            spawn_orientation,
            spawn_scale,
            spawn_material,
            None,
            true,
            None,
            start,
        ),
        (class, archetype) => {
            return Err(format!(
                "spawn spec for {:?} is invalid (class={:?}, archetype={:?})",
                kind, class, archetype
            ));
        }
    };
    Ok(entity)
}

fn handle_console_spawn_command(
    state: &SharedState,
    client_id: u64,
    args: &[&str],
    start: Instant,
) -> Result<(), String> {
    if args.is_empty() {
        return Err(spawn_usage_string());
    }
    let Some(spec) = spawnable_entity_spec_for_token(args[0]) else {
        let available = SPAWNABLE_ENTITY_SPECS
            .iter()
            .map(|spec| spec.canonical_name)
            .collect::<Vec<_>>()
            .join(", ");
        return Err(format!(
            "unknown spawn kind '{}'; available kinds: {available}",
            args[0]
        ));
    };

    let (default_position, default_orientation) = {
        let guard = state.lock().expect("server state lock poisoned");
        default_spawn_pose_for_client(&guard, client_id)
    };
    let parse_usage = || spawn_usage_string();
    let (position, material_id) = match args.len() {
        1 => (default_position, spec.default_material),
        2 => {
            let Some(material_id) = parse_spawn_material_id(args[1]) else {
                return Err(format!("unknown material '{}'", args[1]));
            };
            (default_position, material_id)
        }
        5 => {
            let Some(position) = parse_spawn_vec4(&args[1..5]) else {
                return Err(parse_usage());
            };
            (position, spec.default_material)
        }
        6 => {
            let Some(position) = parse_spawn_vec4(&args[1..5]) else {
                return Err(parse_usage());
            };
            let Some(material_id) = parse_spawn_material_id(args[5]) else {
                return Err(format!("unknown material '{}'", args[5]));
            };
            (position, material_id)
        }
        _ => return Err(parse_usage()),
    };

    let _ = spawn_entity_from_request(
        state,
        spec.kind,
        position,
        default_orientation,
        spec.default_scale,
        material_id,
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
    near_chunk_radius: i32,
    mid_chunk_radius: i32,
    far_chunk_radius: i32,
    start: Instant,
) {
    let message_cpu_start = Instant::now();
    match message {
        ClientMessage::Hello { name } => {
            let (snapshot, _spawned_now) =
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
                            non_empty_chunks: guard.world.non_empty_chunk_count(),
                        }
                    },
                },
            );
            let center_chunk = world_chunk_from_position(snapshot.position);
            sync_streamed_chunks_for_client(
                state,
                client_id,
                center_chunk,
                near_chunk_radius,
                mid_chunk_radius,
                far_chunk_radius,
                true,
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
            let center_chunk = world_chunk_from_position(safe_position);
            sync_streamed_chunks_for_client(
                state,
                client_id,
                center_chunk,
                near_chunk_radius,
                mid_chunk_radius,
                far_chunk_radius,
                false,
            );
        }
        ClientMessage::SetVoxel { position, material } => {
            let requested_voxel = VoxelType(material);
            let maybe_update = {
                let mut guard = state.lock().expect("server state lock poisoned");
                apply_authoritative_voxel_edit(&mut guard, position, requested_voxel)
            };

            if let Some(changed_chunk) = maybe_update {
                force_sync_streamed_clients_for_changed_chunks(
                    state,
                    &[changed_chunk],
                    Some(client_id),
                    near_chunk_radius,
                    mid_chunk_radius,
                    far_chunk_radius,
                );
            }
        }
        ClientMessage::SpawnEntity {
            kind,
            position,
            orientation,
            scale,
            material,
        } => {
            if let Err(message) = spawn_entity_from_request(
                state,
                kind,
                position,
                orientation,
                scale,
                material,
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
        ClientMessage::WorldSubtreeRequest { bounds } => {
            send_world_subtree_patch_to_client(state, client_id, bounds);
        }
        ClientMessage::Ping { nonce } => {
            send_to_client(state, client_id, ServerMessage::Pong { nonce });
        }
    }
    record_server_cpu_sample(state, Some(message_cpu_start.elapsed()), None);
}

pub(super) fn spawn_client_thread(
    stream: TcpStream,
    state: SharedState,
    tick_hz: f32,
    near_chunk_radius: i32,
    mid_chunk_radius: i32,
    far_chunk_radius: i32,
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
            let Ok(encoded) = postcard::to_stdvec(&message) else {
                continue;
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
                    handle_message(
                        &state,
                        client_id,
                        message,
                        tick_hz,
                        near_chunk_radius,
                        mid_chunk_radius,
                        far_chunk_radius,
                        start,
                    );
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
    kind: EntityKind,
    position: [f32; 4],
    orientation: [f32; 4],
    scale: f32,
    material: u8,
    display_name: Option<String>,
    persistent: bool,
    persisted_entity_id: Option<EntityId>,
    start: Instant,
) -> EntitySnapshot {
    let mut guard = state.lock().expect("server state lock poisoned");
    let allocated_id = allocate_or_reserve_server_object_id(&mut guard, persisted_entity_id);
    let now_ms = monotonic_ms(start);
    let entity_id = guard.entity_store.spawn(
        allocated_id,
        EntityClass::Accent,
        kind,
        position,
        orientation,
        scale,
        material,
        now_ms,
    );
    upsert_entity_record(
        &mut guard,
        entity_id,
        EntityClass::Accent,
        None,
        display_name,
        persistent,
        now_ms,
    );
    guard
        .entity_store
        .snapshot(entity_id)
        .expect("spawned entity should exist in store")
}

fn spawn_mob_entity(
    state: &SharedState,
    kind: EntityKind,
    archetype: MobArchetype,
    position: [f32; 4],
    orientation: [f32; 4],
    scale: f32,
    material: u8,
    display_name: Option<String>,
    persistent: bool,
    persisted_mob: Option<PersistedMobEntry>,
    persisted_entity_id: Option<EntityId>,
    start: Instant,
) -> EntitySnapshot {
    let mut guard = state.lock().expect("server state lock poisoned");
    let allocated_id = allocate_or_reserve_server_object_id(&mut guard, persisted_entity_id);
    let now_ms = monotonic_ms(start);
    let entity_id = guard.entity_store.spawn(
        allocated_id,
        EntityClass::Mob,
        kind,
        position,
        orientation,
        scale,
        material,
        now_ms,
    );
    let (default_speed, default_distance, default_tangent) = mob_archetype_defaults(archetype);
    let phase_offset = persisted_mob
        .as_ref()
        .map(|mob| mob.phase_offset)
        .unwrap_or_else(|| ((entity_id as f32) * 0.73).rem_euclid(std::f32::consts::TAU));
    let move_speed = persisted_mob
        .as_ref()
        .map(|mob| mob.move_speed)
        .unwrap_or(default_speed)
        .max(0.1);
    let preferred_distance = persisted_mob
        .as_ref()
        .map(|mob| mob.preferred_distance)
        .unwrap_or(default_distance)
        .max(0.1);
    let tangent_weight = persisted_mob
        .as_ref()
        .map(|mob| mob.tangent_weight)
        .unwrap_or(default_tangent)
        .clamp(0.0, 2.0);
    let initial_phase_tick_seed = (entity_id as u32).wrapping_mul(7477);
    let next_phase_ms =
        phase_spider_next_phase_deadline(now_ms, phase_offset, initial_phase_tick_seed);
    guard.mobs.insert(
        entity_id,
        MobState {
            entity_id,
            archetype,
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
        entity_id,
        EntityClass::Mob,
        None,
        display_name,
        persistent,
        now_ms,
    );
    guard
        .entity_store
        .snapshot(entity_id)
        .expect("spawned mob entity should exist in store")
}
