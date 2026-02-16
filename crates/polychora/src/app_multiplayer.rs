use super::*;
use polychora::shared::worldfield::{
    collect_non_empty_chunks_from_core_in_bounds, Aabb4i, ChunkPayload, RegionClockPrecondition,
    RegionClockUpdate, RegionId, RegionResyncEntry, RegionResyncRequest, RegionSubtreePatch,
    REGION_TILE_EDGE_CHUNKS,
};

const MULTIPLAYER_REGION_RESYNC_MAX_REGIONS: usize = 512;

fn region_id_to_clock_key(region_id: RegionId) -> [i32; 4] {
    [region_id.x, region_id.y, region_id.z, region_id.w]
}

fn apply_region_clock_updates<I>(clocks: &mut HashMap<[i32; 4], u64>, updates: I) -> usize
where
    I: IntoIterator<Item = ([i32; 4], u64)>,
{
    let mut regressed = 0usize;
    for (region_id, clock) in updates {
        match clocks.get_mut(&region_id) {
            Some(existing) if clock < *existing => {
                regressed = regressed.saturating_add(1);
            }
            Some(existing) => {
                *existing = clock;
            }
            None => {
                clocks.insert(region_id, clock);
            }
        }
    }
    regressed
}

fn region_clock_precondition_mismatch_count(
    clocks: &HashMap<[i32; 4], u64>,
    preconditions: &[RegionClockPrecondition],
) -> usize {
    preconditions
        .iter()
        .filter(|precondition| {
            let key = region_id_to_clock_key(precondition.region_id);
            clocks.get(&key).copied().unwrap_or(0) != precondition.expected_clock
        })
        .count()
}

fn region_resync_entries_for_bounds(
    clocks: &HashMap<[i32; 4], u64>,
    bounds: Aabb4i,
    max_regions: usize,
) -> Vec<RegionResyncEntry> {
    if !bounds.is_valid() || max_regions == 0 {
        return Vec::new();
    }

    let region_min = [
        bounds.min[0].div_euclid(REGION_TILE_EDGE_CHUNKS),
        bounds.min[1].div_euclid(REGION_TILE_EDGE_CHUNKS),
        bounds.min[2].div_euclid(REGION_TILE_EDGE_CHUNKS),
        bounds.min[3].div_euclid(REGION_TILE_EDGE_CHUNKS),
    ];
    let region_max = [
        bounds.max[0].div_euclid(REGION_TILE_EDGE_CHUNKS),
        bounds.max[1].div_euclid(REGION_TILE_EDGE_CHUNKS),
        bounds.max[2].div_euclid(REGION_TILE_EDGE_CHUNKS),
        bounds.max[3].div_euclid(REGION_TILE_EDGE_CHUNKS),
    ];

    let mut entries = Vec::new();
    for rw in region_min[3]..=region_max[3] {
        for rz in region_min[2]..=region_max[2] {
            for ry in region_min[1]..=region_max[1] {
                for rx in region_min[0]..=region_max[0] {
                    if entries.len() >= max_regions {
                        return entries;
                    }
                    let key = [rx, ry, rz, rw];
                    let clock = clocks.get(&key).copied().unwrap_or(0);
                    entries.push(RegionResyncEntry {
                        region_id: RegionId {
                            x: rx,
                            y: ry,
                            z: rz,
                            w: rw,
                        },
                        client_clock: clock,
                    });
                }
            }
        }
    }
    entries
}

fn region_resync_entries_for_mismatched_preconditions(
    clocks: &HashMap<[i32; 4], u64>,
    preconditions: &[RegionClockPrecondition],
    max_regions: usize,
) -> Vec<RegionResyncEntry> {
    if max_regions == 0 {
        return Vec::new();
    }
    let mut entries = Vec::new();
    for precondition in preconditions {
        if entries.len() >= max_regions {
            break;
        }
        let key = region_id_to_clock_key(precondition.region_id);
        let local_clock = clocks.get(&key).copied().unwrap_or(0);
        if local_clock == precondition.expected_clock {
            continue;
        }
        entries.push(RegionResyncEntry {
            region_id: precondition.region_id,
            client_clock: local_clock,
        });
    }
    entries
}

fn sanitize_remote_velocity(mut velocity: [f32; 4], max_speed: f32) -> [f32; 4] {
    if velocity.iter().any(|v| !v.is_finite()) {
        return [0.0; 4];
    }
    let speed = dot4(velocity, velocity).sqrt();
    if speed.is_finite() && speed > max_speed && max_speed > 0.0 {
        let clamp = max_speed / speed;
        for axis in &mut velocity {
            *axis *= clamp;
        }
    }
    velocity
}

fn vec4_is_finite(v: [f32; 4]) -> bool {
    v.iter().all(|axis| axis.is_finite())
}

fn sanitize_remote_position(position: [f32; 4], fallback: [f32; 4]) -> [f32; 4] {
    if vec4_is_finite(position) {
        position
    } else {
        fallback
    }
}

fn sanitize_remote_orientation(orientation: [f32; 4], fallback: [f32; 4]) -> [f32; 4] {
    normalize4_with_fallback(orientation, fallback)
}

fn sanitize_remote_scale(scale: f32, fallback: f32) -> f32 {
    if scale.is_finite() {
        scale
    } else {
        fallback
    }
}

impl App {
    pub(super) fn poll_multiplayer_events(&mut self) {
        loop {
            let event = match self
                .multiplayer
                .as_ref()
                .and_then(|client| client.try_recv())
            {
                Some(event) => event,
                None => break,
            };

            match event {
                MultiplayerEvent::Message(message) => self.handle_multiplayer_message(message),
                MultiplayerEvent::Disconnected(reason) => {
                    eprintln!("Multiplayer disconnected: {reason}");
                    self.reset_multiplayer_connection_state();
                    break;
                }
            }
        }
    }

    pub(super) fn upsert_remote_player_entity(
        &mut self,
        entity: multiplayer::EntitySnapshot,
        received_at: Instant,
    ) {
        let normalized_look = sanitize_remote_orientation(entity.orientation, [0.0, 0.0, 1.0, 0.0]);
        let sanitized_position = sanitize_remote_position(entity.position, [0.0; 4]);
        let player_name = entity.display_name.clone().unwrap_or_else(|| {
            entity
                .owner_client_id
                .map(|id| format!("player-{id}"))
                .unwrap_or_else(|| format!("entity-{}", entity.entity_id))
        });
        if let Some(existing) = self.remote_players.get_mut(&entity.entity_id) {
            let previous_position = existing.position;
            let next_position = sanitize_remote_position(entity.position, existing.position);

            existing.owner_client_id = entity.owner_client_id;
            existing.name = player_name;
            existing.position = next_position;
            existing.look = normalized_look;
            existing.last_update_ms = entity.last_update_ms;
            existing.last_received_at = received_at;
            existing.velocity =
                sanitize_remote_velocity(entity.velocity, REMOTE_PLAYER_MAX_PREDICTED_SPEED);

            if distance4(previous_position, next_position) > REMOTE_PLAYER_TELEPORT_SNAP_DISTANCE {
                existing.render_position = existing.position;
                existing.render_look = existing.look;
                existing.velocity = [0.0; 4];
                existing.footstep_distance_accum = 0.0;
            }
            return;
        }

        self.remote_players.insert(
            entity.entity_id,
            RemotePlayerState {
                owner_client_id: entity.owner_client_id,
                name: player_name,
                position: sanitized_position,
                look: normalized_look,
                last_update_ms: entity.last_update_ms,
                render_position: sanitized_position,
                render_look: normalized_look,
                velocity: sanitize_remote_velocity(
                    entity.velocity,
                    REMOTE_PLAYER_MAX_PREDICTED_SPEED,
                ),
                last_received_at: received_at,
                footstep_distance_accum: 0.0,
            },
        );
    }

    pub(super) fn smooth_remote_players(&mut self, dt: f32, now: Instant) {
        let dt = dt.clamp(0.0, 0.25);
        if dt <= 0.0 {
            return;
        }
        let pos_alpha = 1.0 - (-REMOTE_PLAYER_POSITION_SMOOTH_HZ * dt).exp();
        let look_alpha = 1.0 - (-REMOTE_PLAYER_LOOK_SMOOTH_HZ * dt).exp();
        let listener_position = self.camera.position;
        let mut footstep_scales = Vec::new();
        for player in self.remote_players.values_mut() {
            let previous_render_position = player.render_position;
            let network_age = now.duration_since(player.last_received_at).as_secs_f32();
            let predict_time = (network_age + REMOTE_PLAYER_PREDICTION_LEAD_S)
                .clamp(0.0, REMOTE_PLAYER_MAX_PREDICTION_S);

            let mut predicted_position = player.position;
            for axis in 0..4 {
                predicted_position[axis] += player.velocity[axis] * predict_time;
            }

            let snapped = if distance4(player.render_position, predicted_position)
                > REMOTE_PLAYER_TELEPORT_SNAP_DISTANCE
            {
                player.render_position = predicted_position;
                true
            } else {
                player.render_position =
                    lerp4(player.render_position, predicted_position, pos_alpha);
                false
            };

            player.render_look = normalize4_with_fallback(
                lerp4(player.render_look, player.look, look_alpha),
                player.look,
            );

            if snapped || network_age > REMOTE_FOOTSTEP_MAX_NETWORK_AGE_S {
                player.footstep_distance_accum = 0.0;
                continue;
            }

            let moved_dx = player.render_position[0] - previous_render_position[0];
            let moved_dy = player.render_position[1] - previous_render_position[1];
            let moved_dz = player.render_position[2] - previous_render_position[2];
            let moved_dw = player.render_position[3] - previous_render_position[3];
            let moved_xzw =
                (moved_dx * moved_dx + moved_dz * moved_dz + moved_dw * moved_dw).sqrt();
            let moved_speed_xzw = moved_xzw / dt;
            let moved_speed_y = moved_dy.abs() / dt;

            if moved_speed_xzw <= REMOTE_FOOTSTEP_MIN_XZW_SPEED
                || moved_speed_y > REMOTE_FOOTSTEP_MAX_VERTICAL_SPEED
            {
                player.footstep_distance_accum = 0.0;
                continue;
            }

            player.footstep_distance_accum += moved_xzw;
            while player.footstep_distance_accum >= FOOTSTEP_DISTANCE_WALK {
                if footstep_scales.len() >= REMOTE_FOOTSTEP_MAX_PER_FRAME {
                    player.footstep_distance_accum = 0.0;
                    break;
                }

                let distance_to_listener = distance4(listener_position, player.render_position);
                if distance_to_listener < REMOTE_FOOTSTEP_MAX_DISTANCE {
                    let distance_t =
                        1.0 - (distance_to_listener / REMOTE_FOOTSTEP_MAX_DISTANCE).clamp(0.0, 1.0);
                    let distance_gain = distance_t * distance_t;
                    let speed_gain =
                        (moved_speed_xzw / REMOTE_PLAYER_MAX_PREDICTED_SPEED).clamp(0.30, 1.0);
                    let gain = (distance_gain * speed_gain).clamp(0.05, 0.65);
                    footstep_scales.push(gain);
                }
                player.footstep_distance_accum -= FOOTSTEP_DISTANCE_WALK;
            }
        }

        for scale in footstep_scales {
            self.audio.play_scaled(SoundEffect::Footstep, scale);
        }
    }

    pub(super) fn smooth_remote_entities(&mut self, dt: f32) {
        let dt = dt.clamp(0.0, 0.25);
        if dt <= 0.0 {
            return;
        }
        let pos_alpha = 1.0 - (-REMOTE_ENTITY_POSITION_SMOOTH_HZ * dt).exp();
        let orientation_alpha = 1.0 - (-REMOTE_ENTITY_ORIENTATION_SMOOTH_HZ * dt).exp();
        for entity in self.remote_entities.values_mut() {
            let mut predicted_position = entity.position;
            for axis in 0..4 {
                predicted_position[axis] += entity.velocity[axis] * dt;
            }
            if distance4(entity.render_position, predicted_position)
                > REMOTE_ENTITY_TELEPORT_SNAP_DISTANCE
            {
                entity.render_position = predicted_position;
                entity.render_orientation = entity.orientation;
            } else {
                entity.render_position =
                    lerp4(entity.render_position, predicted_position, pos_alpha);
                entity.render_orientation = normalize4_with_fallback(
                    lerp4(
                        entity.render_orientation,
                        entity.orientation,
                        orientation_alpha,
                    ),
                    entity.orientation,
                );
            }
        }
    }

    pub(super) fn acknowledge_pending_voxel_edit(
        &mut self,
        client_edit_id: Option<u64>,
        position: [i32; 4],
        material: u8,
    ) {
        let index = if let Some(edit_id) = client_edit_id {
            self.pending_voxel_edits
                .iter()
                .position(|entry| entry.client_edit_id == edit_id)
        } else {
            self.pending_voxel_edits
                .iter()
                .position(|entry| entry.position == position && entry.material == material)
                .or_else(|| {
                    self.pending_voxel_edits
                        .iter()
                        .position(|entry| entry.position == position)
                })
        };
        if let Some(index) = index {
            let _ = self.pending_voxel_edits.remove(index);
        }
    }

    pub(super) fn reapply_pending_voxel_edits(&mut self, now: Instant) {
        self.pending_voxel_edits.retain(|entry| {
            now.saturating_duration_since(entry.created_at) <= MULTIPLAYER_PENDING_EDIT_TIMEOUT
        });
        for entry in &self.pending_voxel_edits {
            self.scene.world.set_voxel(
                entry.position[0],
                entry.position[1],
                entry.position[2],
                entry.position[3],
                voxel::VoxelType(entry.material),
            );
        }
    }

    pub(super) fn queue_pending_player_movement_modifier(
        &mut self,
        delta_position: [f32; 4],
        delta_velocity_y: f32,
        source_entity_id: Option<u64>,
    ) {
        if delta_position.iter().any(|axis| !axis.is_finite()) || !delta_velocity_y.is_finite() {
            eprintln!(
                "[impulse-debug][client] drop invalid modifier source_entity={:?} delta={:?} delta_velocity_y={}",
                source_entity_id, delta_position, delta_velocity_y
            );
            return;
        }
        let mut clamped_delta = delta_position;
        let delta_len_sq = dot4(clamped_delta, clamped_delta);
        if delta_len_sq.is_finite() && delta_len_sq > 1e-6 {
            let delta_len = delta_len_sq.sqrt();
            if delta_len > MULTIPLAYER_PLAYER_MODIFIER_MAX_TRANSLATION {
                let clamp = MULTIPLAYER_PLAYER_MODIFIER_MAX_TRANSLATION / delta_len;
                for axis in &mut clamped_delta {
                    *axis *= clamp;
                }
            }
        }
        let raw_len = dot4(delta_position, delta_position).sqrt();
        let clamped_len = dot4(clamped_delta, clamped_delta).sqrt();
        self.pending_player_movement_modifiers
            .push_back(PendingPlayerMovementModifier {
                delta_position: clamped_delta,
                delta_velocity_y: delta_velocity_y.clamp(
                    -MULTIPLAYER_PLAYER_MODIFIER_MAX_VELOCITY_Y_DELTA,
                    MULTIPLAYER_PLAYER_MODIFIER_MAX_VELOCITY_Y_DELTA,
                ),
                source_entity_id,
            });
        eprintln!(
            "[impulse-debug][client] queue modifier source_entity={:?} raw_delta={:?} raw_len={:.3} clamped_delta={:?} clamped_len={:.3} raw_dvy={:.3} clamped_dvy={:.3} queue_len={}",
            source_entity_id,
            delta_position,
            raw_len,
            clamped_delta,
            clamped_len,
            delta_velocity_y,
            self.pending_player_movement_modifiers
                .back()
                .map(|m| m.delta_velocity_y)
                .unwrap_or(0.0),
            self.pending_player_movement_modifiers.len()
        );
        if self.pending_player_movement_modifiers.len() > MULTIPLAYER_PENDING_PLAYER_MODIFIER_MAX {
            let overflow = self.pending_player_movement_modifiers.len()
                - MULTIPLAYER_PENDING_PLAYER_MODIFIER_MAX;
            self.pending_player_movement_modifiers.drain(0..overflow);
        }
    }

    pub(super) fn apply_pending_player_movement_modifiers(&mut self) {
        while let Some(modifier) = self.pending_player_movement_modifiers.pop_front() {
            let mut added_velocity = [0.0f32; 4];
            for axis in [0usize, 2, 3] {
                added_velocity[axis] =
                    modifier.delta_position[axis] * MULTIPLAYER_PLAYER_MODIFIER_IMPULSE_GAIN_XZW;
                self.player_modifier_external_velocity[axis] += added_velocity[axis];
            }
            let xzw_speed = (self.player_modifier_external_velocity[0]
                * self.player_modifier_external_velocity[0]
                + self.player_modifier_external_velocity[2]
                    * self.player_modifier_external_velocity[2]
                + self.player_modifier_external_velocity[3]
                    * self.player_modifier_external_velocity[3])
                .sqrt();
            if xzw_speed > MULTIPLAYER_PLAYER_MODIFIER_MAX_EXTERNAL_SPEED_XZW {
                let clamp = MULTIPLAYER_PLAYER_MODIFIER_MAX_EXTERNAL_SPEED_XZW / xzw_speed;
                self.player_modifier_external_velocity[0] *= clamp;
                self.player_modifier_external_velocity[2] *= clamp;
                self.player_modifier_external_velocity[3] *= clamp;
            }
            let previous_velocity_y = self.camera.velocity_y;
            if !self.camera.is_flying && modifier.delta_velocity_y.abs() > 1e-5 {
                self.camera.velocity_y += modifier.delta_velocity_y;
                if modifier.delta_velocity_y > 0.0 {
                    self.camera.is_grounded = false;
                }
            }
            eprintln!(
                "[impulse-debug][client] apply modifier source_entity={:?} delta={:?} added_external_velocity={:?} external_velocity_now={:?} prev_vy={:.3} next_vy={:.3}",
                modifier.source_entity_id,
                modifier.delta_position,
                added_velocity,
                self.player_modifier_external_velocity,
                previous_velocity_y,
                self.camera.velocity_y
            );
        }
    }

    pub(super) fn apply_multiplayer_chunk_batch(
        &mut self,
        revision: u64,
        chunks: Vec<multiplayer::WorldChunkPayload>,
    ) -> usize {
        let mut applied_chunks = 0usize;

        for payload in chunks {
            if payload.voxels.len() != voxel::CHUNK_VOLUME {
                eprintln!(
                    "Ignoring malformed multiplayer chunk payload at ({}, {}, {}, {}): expected {} voxels, got {}",
                    payload.chunk_pos[0],
                    payload.chunk_pos[1],
                    payload.chunk_pos[2],
                    payload.chunk_pos[3],
                    voxel::CHUNK_VOLUME,
                    payload.voxels.len()
                );
                continue;
            }

            let mut voxels = Box::new([voxel::VoxelType::AIR; voxel::CHUNK_VOLUME]);
            let mut solid_count = 0u32;
            for (idx, material) in payload.voxels.into_iter().enumerate() {
                let voxel_type = voxel::VoxelType(material);
                if voxel_type.is_solid() {
                    solid_count += 1;
                }
                voxels[idx] = voxel_type;
            }

            let chunk = voxel::chunk::Chunk {
                voxels,
                solid_count,
                dirty: true,
            };
            let chunk_pos = voxel::ChunkPos::new(
                payload.chunk_pos[0],
                payload.chunk_pos[1],
                payload.chunk_pos[2],
                payload.chunk_pos[3],
            );
            self.scene
                .insert_lod_chunk(payload.lod_level, chunk_pos, chunk);
            applied_chunks += 1;
        }

        if applied_chunks > 0 {
            eprintln!(
                "Applied multiplayer chunk batch rev={} chunks={}",
                revision, applied_chunks
            );
        }
        applied_chunks
    }

    pub(super) fn apply_multiplayer_chunk_unload_batch(
        &mut self,
        revision: u64,
        chunks: Vec<multiplayer::WorldChunkCoordPayload>,
    ) {
        let mut removed_chunks = 0usize;
        for payload in chunks {
            let pos = voxel::ChunkPos::new(
                payload.chunk_pos[0],
                payload.chunk_pos[1],
                payload.chunk_pos[2],
                payload.chunk_pos[3],
            );
            if self.scene.remove_lod_chunk(payload.lod_level, pos) {
                removed_chunks += 1;
            }
        }
        if removed_chunks > 0 {
            eprintln!(
                "Applied multiplayer chunk unload batch rev={} chunks={}",
                revision, removed_chunks
            );
        }
    }

    fn request_multiplayer_region_resync(
        &mut self,
        mut regions: Vec<RegionResyncEntry>,
        reason: &str,
    ) {
        if regions.is_empty() {
            return;
        }
        let now = Instant::now();
        if now.duration_since(self.multiplayer_last_region_resync_request)
            < MULTIPLAYER_REGION_RESYNC_MIN_INTERVAL
        {
            return;
        }
        regions.sort_unstable_by_key(|entry| {
            (
                entry.region_id.w,
                entry.region_id.z,
                entry.region_id.y,
                entry.region_id.x,
            )
        });
        regions.dedup_by_key(|entry| entry.region_id);
        if regions.len() > MULTIPLAYER_REGION_RESYNC_MAX_REGIONS {
            regions.truncate(MULTIPLAYER_REGION_RESYNC_MAX_REGIONS);
        }
        let region_count = regions.len();

        if let Some(client) = self.multiplayer.as_ref() {
            self.multiplayer_last_region_resync_request = now;
            client.send(MultiplayerClientMessage::WorldRegionResyncRequest {
                request: RegionResyncRequest { regions },
            });
            eprintln!(
                "Requested multiplayer region resync: reason={reason} regions={region_count}"
            );
        }
    }

    fn request_multiplayer_region_resync_for_bounds(&mut self, bounds: Aabb4i, reason: &str) {
        let regions = region_resync_entries_for_bounds(
            &self.multiplayer_region_clocks,
            bounds,
            MULTIPLAYER_REGION_RESYNC_MAX_REGIONS,
        );
        self.request_multiplayer_region_resync(regions, reason);
    }

    pub(super) fn apply_multiplayer_region_patch(
        &mut self,
        patch: RegionSubtreePatch,
    ) -> Option<(usize, usize)> {
        if let Some(last_seq) = self.multiplayer_last_region_patch_seq {
            if patch.patch_seq <= last_seq {
                eprintln!(
                    "Ignoring stale multiplayer region patch seq={} last_seq={}",
                    patch.patch_seq, last_seq
                );
                return None;
            }
            let expected_next = last_seq.wrapping_add(1).max(1);
            if patch.patch_seq != expected_next {
                eprintln!(
                    "Rejecting out-of-order multiplayer region patch seq={} expected_next={}",
                    patch.patch_seq, expected_next
                );
                self.request_multiplayer_region_resync_for_bounds(
                    patch.bounds,
                    "out_of_order_patch_seq",
                );
                return None;
            }
        }

        let mismatches = region_clock_precondition_mismatch_count(
            &self.multiplayer_region_clocks,
            &patch.preconditions,
        );
        if mismatches > 0 {
            eprintln!(
                "Rejecting multiplayer region patch seq={} due to {} clock precondition mismatches",
                patch.patch_seq, mismatches
            );
            let mut regions = region_resync_entries_for_mismatched_preconditions(
                &self.multiplayer_region_clocks,
                &patch.preconditions,
                MULTIPLAYER_REGION_RESYNC_MAX_REGIONS,
            );
            if regions.is_empty() {
                regions = region_resync_entries_for_bounds(
                    &self.multiplayer_region_clocks,
                    patch.bounds,
                    MULTIPLAYER_REGION_RESYNC_MAX_REGIONS,
                );
            }
            self.request_multiplayer_region_resync(regions, "clock_precondition_mismatch");
            return None;
        }

        let desired_chunks: HashMap<voxel::ChunkPos, ChunkPayload> =
            collect_non_empty_chunks_from_core_in_bounds(&patch.subtree, patch.bounds)
                .into_iter()
                .map(|(key, payload)| (key.to_chunk_pos(), payload))
                .collect();

        let mut current_chunks = Vec::new();
        self.scene.world.gather_non_empty_chunks_in_bounds(
            patch.bounds.min,
            patch.bounds.max,
            &mut current_chunks,
        );

        let mut removed_chunks = 0usize;
        for chunk_pos in current_chunks {
            if !desired_chunks.contains_key(&chunk_pos)
                && self
                    .scene
                    .remove_lod_chunk(scene::VOXEL_LOD_LEVEL_NEAR, chunk_pos)
            {
                removed_chunks += 1;
            }
        }

        let mut applied_chunks = 0usize;
        for (chunk_pos, payload) in desired_chunks {
            let chunk = match payload.to_voxel_chunk() {
                Ok(chunk) => chunk,
                Err(error) => {
                    eprintln!(
                        "Ignoring malformed region patch payload at ({}, {}, {}, {}): {}",
                        chunk_pos.x, chunk_pos.y, chunk_pos.z, chunk_pos.w, error
                    );
                    continue;
                }
            };
            self.scene
                .insert_lod_chunk(scene::VOXEL_LOD_LEVEL_NEAR, chunk_pos, chunk);
            applied_chunks += 1;
        }

        let regressed = apply_region_clock_updates(
            &mut self.multiplayer_region_clocks,
            patch
                .clock_updates
                .iter()
                .map(|update: &RegionClockUpdate| {
                    (region_id_to_clock_key(update.region_id), update.new_clock)
                }),
        );
        if regressed > 0 {
            eprintln!(
                "Ignored {} regressed multiplayer region clock updates from region patch",
                regressed
            );
        }

        self.multiplayer_last_region_patch_seq = Some(patch.patch_seq);
        eprintln!(
            "Applied multiplayer region patch seq={} bounds={:?}->{:?} upserts={} removals={}",
            patch.patch_seq, patch.bounds.min, patch.bounds.max, applied_chunks, removed_chunks
        );
        Some((applied_chunks, removed_chunks))
    }

    pub(super) fn handle_multiplayer_message(&mut self, message: multiplayer::ServerMessage) {
        let received_at = Instant::now();
        match message {
            multiplayer::ServerMessage::Welcome {
                client_id,
                tick_hz,
                world,
                ..
            } => {
                self.multiplayer_self_id = Some(client_id);
                self.multiplayer_region_clocks.clear();
                self.multiplayer_last_region_patch_seq = None;
                self.next_multiplayer_edit_id = 1;
                self.pending_voxel_edits.clear();
                self.pending_player_movement_modifiers.clear();
                self.player_modifier_external_velocity = [0.0; 4];
                eprintln!(
                    "Multiplayer connected: client_id={} world_rev={} chunks={} server_tick_hz={:.2}",
                    client_id, world.revision, world.non_empty_chunks, tick_hz
                );
                // Note: server already sends WorldSnapshot + streamed chunks
                // during Hello processing, so no need to request again here.
                // A redundant RequestWorldSnapshot would wipe procgen chunks.
            }
            multiplayer::ServerMessage::Error { message } => {
                eprintln!("Multiplayer server error: {message}");
                self.append_dev_console_log_line(format!("[server] {message}"));
            }
            multiplayer::ServerMessage::WorldVoxelSet {
                position,
                material,
                source_client_id,
                client_edit_id,
                ..
            } => {
                self.scene.world.set_voxel(
                    position[0],
                    position[1],
                    position[2],
                    position[3],
                    voxel::VoxelType(material),
                );
                let from_self = source_client_id == self.multiplayer_self_id;
                if from_self {
                    self.acknowledge_pending_voxel_edit(client_edit_id, position, material);
                } else if source_client_id.is_some() {
                    if material == voxel::VoxelType::AIR.0 {
                        self.audio.play(SoundEffect::Break);
                    } else {
                        self.audio.play(SoundEffect::Place);
                    }
                }
            }
            multiplayer::ServerMessage::WorldSnapshot { world } => {
                let mut cursor = &world.bytes[..];
                match voxel::io::load_world(&mut cursor) {
                    Ok(new_world) => {
                        self.scene.replace_world(new_world);
                        self.multiplayer_region_clocks.clear();
                        self.multiplayer_last_region_patch_seq = None;
                        eprintln!(
                            "Applied multiplayer world snapshot rev={} chunks={}",
                            world.revision, world.non_empty_chunks
                        );
                        // Warm near chunks so the first drawn frame has local payloads ready.
                        self.scene.preload_spawn_chunks(
                            self.camera.position,
                            self.vte_lod_near_max_distance,
                        );
                        self.world_ready = true;
                        eprintln!("World ready: snapshot applied");
                    }
                    Err(error) => {
                        eprintln!("Failed to load multiplayer world snapshot: {error}");
                    }
                }
            }
            multiplayer::ServerMessage::WorldChunkBatch { revision, chunks } => {
                let applied_chunks = self.apply_multiplayer_chunk_batch(revision, chunks);
                if applied_chunks > 0 && !self.world_ready {
                    self.world_ready = true;
                    eprintln!("World ready: streamed chunk data applied");
                }
            }
            multiplayer::ServerMessage::WorldChunkUnloadBatch { revision, chunks } => {
                self.apply_multiplayer_chunk_unload_batch(revision, chunks);
            }
            multiplayer::ServerMessage::WorldRegionClockUpdate { updates } => {
                let regressed = apply_region_clock_updates(
                    &mut self.multiplayer_region_clocks,
                    updates
                        .into_iter()
                        .map(|update| (update.region_id, update.clock)),
                );
                if regressed > 0 {
                    eprintln!(
                        "Ignored {} regressed multiplayer region clock updates",
                        regressed
                    );
                }
            }
            multiplayer::ServerMessage::WorldRegionPatch { patch } => {
                if self.apply_multiplayer_region_patch(patch).is_some() && !self.world_ready {
                    self.world_ready = true;
                    eprintln!("World ready: region patch applied");
                }
            }
            multiplayer::ServerMessage::Pong { .. } => {}
            multiplayer::ServerMessage::EntitySpawned { entity } => match entity.class {
                multiplayer::EntityClass::Player => {
                    self.remote_entities.remove(&entity.entity_id);
                    if entity.owner_client_id == self.multiplayer_self_id {
                        self.remote_players.remove(&entity.entity_id);
                    } else {
                        self.upsert_remote_player_entity(entity, received_at);
                    }
                }
                multiplayer::EntityClass::Accent | multiplayer::EntityClass::Mob => {
                    let fallback_orientation = [0.0, 0.0, 1.0, 0.0];
                    let sanitized_position = sanitize_remote_position(entity.position, [0.0; 4]);
                    let sanitized_orientation =
                        sanitize_remote_orientation(entity.orientation, fallback_orientation);
                    let sanitized_scale = sanitize_remote_scale(entity.scale, 1.0);
                    self.remote_entities.insert(
                        entity.entity_id,
                        RemoteEntityState {
                            kind: entity.kind,
                            position: sanitized_position,
                            orientation: sanitized_orientation,
                            velocity: sanitize_remote_velocity(
                                entity.velocity,
                                REMOTE_PLAYER_MAX_PREDICTED_SPEED,
                            ),
                            scale: sanitized_scale,
                            material: entity.material,
                            render_position: sanitized_position,
                            render_orientation: sanitized_orientation,
                            last_received_at: received_at,
                        },
                    );
                }
            },
            multiplayer::ServerMessage::EntityDestroyed { entity_id } => {
                self.remote_players.remove(&entity_id);
                self.remote_entities.remove(&entity_id);
            }
            multiplayer::ServerMessage::EntityTransforms { entities, .. } => {
                for transform in entities {
                    if let Some(player) = self.remote_players.get_mut(&transform.entity_id) {
                        let previous_position = player.position;
                        let next_position =
                            sanitize_remote_position(transform.position, player.position);
                        let normalized_look =
                            sanitize_remote_orientation(transform.orientation, player.look);
                        player.position = next_position;
                        player.look = normalized_look;
                        player.last_update_ms = transform.last_update_ms;
                        player.last_received_at = received_at;
                        player.velocity = sanitize_remote_velocity(
                            transform.velocity,
                            REMOTE_PLAYER_MAX_PREDICTED_SPEED,
                        );
                        if distance4(previous_position, next_position)
                            > REMOTE_PLAYER_TELEPORT_SNAP_DISTANCE
                        {
                            player.render_position = player.position;
                            player.render_look = player.look;
                            player.velocity = [0.0; 4];
                            player.footstep_distance_accum = 0.0;
                        }
                        continue;
                    }
                    if let Some(existing) = self.remote_entities.get_mut(&transform.entity_id) {
                        existing.position =
                            sanitize_remote_position(transform.position, existing.position);
                        existing.orientation = sanitize_remote_orientation(
                            transform.orientation,
                            existing.orientation,
                        );
                        existing.velocity = sanitize_remote_velocity(
                            transform.velocity,
                            REMOTE_PLAYER_MAX_PREDICTED_SPEED,
                        );
                        existing.scale = sanitize_remote_scale(transform.scale, existing.scale);
                        existing.material = transform.material;
                        existing.last_received_at = received_at;
                    }
                }
            }
            multiplayer::ServerMessage::Explosion {
                position, radius, ..
            } => {
                let max_distance = 40.0f32;
                let distance = distance4(self.camera.position, position);
                if distance <= max_distance {
                    let distance_t = 1.0 - (distance / max_distance).clamp(0.0, 1.0);
                    let distance_gain = distance_t * distance_t;
                    let radius_gain = (radius / 2.5).clamp(0.6, 1.6);
                    self.audio
                        .play_scaled(SoundEffect::Break, distance_gain * radius_gain * 1.4);
                }
            }
            multiplayer::ServerMessage::PlayerMovementModifier {
                delta_position,
                delta_velocity_y,
                source_entity_id,
            } => {
                eprintln!(
                    "[impulse-debug][client] recv modifier source_entity={:?} delta={:?} delta_velocity_y={:.3}",
                    source_entity_id,
                    delta_position,
                    delta_velocity_y
                );
                self.queue_pending_player_movement_modifier(
                    delta_position,
                    delta_velocity_y,
                    source_entity_id,
                );
            }
        }
    }

    pub(super) fn send_multiplayer_player_update(&mut self, now: Instant, look_dir: [f32; 4]) {
        if now.duration_since(self.last_multiplayer_player_update)
            < MULTIPLAYER_PLAYER_UPDATE_INTERVAL
        {
            return;
        }
        self.last_multiplayer_player_update = now;
        if let Some(client) = self.multiplayer.as_ref() {
            client.send(MultiplayerClientMessage::UpdatePlayer {
                position: self.camera.position,
                look: look_dir,
            });
        }
    }

    pub(super) fn send_multiplayer_voxel_update(
        &mut self,
        now: Instant,
        position: [i32; 4],
        material: u8,
    ) {
        if self.multiplayer.is_none() {
            return;
        }

        let client_edit_id = self.next_multiplayer_edit_id;
        self.next_multiplayer_edit_id = self.next_multiplayer_edit_id.wrapping_add(1).max(1);
        self.pending_voxel_edits.push(PendingVoxelEdit {
            client_edit_id,
            position,
            material,
            created_at: now,
        });

        if self.pending_voxel_edits.len() > MULTIPLAYER_PENDING_EDIT_MAX {
            let overflow = self.pending_voxel_edits.len() - MULTIPLAYER_PENDING_EDIT_MAX;
            self.pending_voxel_edits.drain(0..overflow);
        }

        if let Some(client) = self.multiplayer.as_ref() {
            client.send(MultiplayerClientMessage::SetVoxel {
                position,
                material,
                client_edit_id: Some(client_edit_id),
            });
        }
    }

    pub(super) fn send_multiplayer_console_command(&self, command: &str) -> bool {
        let Some(client) = self.multiplayer.as_ref() else {
            return false;
        };

        client.send(MultiplayerClientMessage::ConsoleCommand {
            command: command.trim().to_string(),
        });
        true
    }

    pub(super) fn remote_player_instances(&self, time_s: f32) -> Vec<common::ModelInstance> {
        let mut ids: Vec<u64> = self.remote_players.keys().copied().collect();
        ids.sort_unstable();
        let mut instances = Vec::with_capacity(ids.len() * REMOTE_AVATAR_PART_COUNT_ESTIMATE);
        for entity_id in ids {
            if let Some(player) = self.remote_players.get(&entity_id) {
                if !vec4_is_finite(player.render_position) || !vec4_is_finite(player.render_look) {
                    continue;
                }
                let avatar_seed_id = player.owner_client_id.unwrap_or(entity_id);
                instances.extend(build_remote_player_avatar_instances(
                    avatar_seed_id,
                    stable_name_hash(&player.name),
                    player.render_position,
                    player.render_look,
                    time_s,
                ));
            }
        }
        instances
    }

    pub(super) fn remote_entity_instances(&self) -> Vec<common::ModelInstance> {
        let mut ids: Vec<u64> = self.remote_entities.keys().copied().collect();
        ids.sort_unstable();
        let mut instances = Vec::with_capacity(ids.len() * 8);
        for entity_id in ids {
            if let Some(entity) = self.remote_entities.get(&entity_id) {
                if !vec4_is_finite(entity.render_position)
                    || !vec4_is_finite(entity.render_orientation)
                    || !entity.scale.is_finite()
                {
                    continue;
                }
                match entity.kind {
                    multiplayer::EntityKind::PlayerAvatar => {}
                    multiplayer::EntityKind::TestCube => {
                        let basis = orthonormal_basis_from_forward(entity.render_orientation);
                        instances.push(build_centered_model_instance(
                            entity.render_position,
                            &basis,
                            [entity.scale; 4],
                            [entity.material as u32; 8],
                        ));
                    }
                    multiplayer::EntityKind::TestRotor => {
                        let basis = orthonormal_basis_from_forward(entity.render_orientation);
                        instances.push(build_centered_model_instance(
                            entity.render_position,
                            &basis,
                            [
                                entity.scale * 0.56,
                                entity.scale * 0.56,
                                entity.scale * 1.35,
                                entity.scale * 0.82,
                            ],
                            [
                                entity.material as u32,
                                entity.material as u32,
                                entity.material as u32,
                                (entity.material.saturating_add(1)) as u32,
                                entity.material as u32,
                                (entity.material.saturating_add(1)) as u32,
                                entity.material as u32,
                                (entity.material.saturating_add(1)) as u32,
                            ],
                        ));
                    }
                    multiplayer::EntityKind::TestDrifter => {
                        let basis = orthonormal_basis_from_forward(entity.render_orientation);
                        instances.push(build_centered_model_instance(
                            entity.render_position,
                            &basis,
                            [
                                entity.scale * 1.15,
                                entity.scale * 0.44,
                                entity.scale * 0.72,
                                entity.scale * 1.05,
                            ],
                            [
                                (entity.material.saturating_add(2)) as u32,
                                entity.material as u32,
                                entity.material as u32,
                                entity.material as u32,
                                entity.material as u32,
                                (entity.material.saturating_add(2)) as u32,
                                entity.material as u32,
                                entity.material as u32,
                            ],
                        ));
                    }
                    multiplayer::EntityKind::MobSeeker => {
                        let basis = orthonormal_basis_from_forward(entity.render_orientation);
                        instances.push(build_centered_model_instance(
                            entity.render_position,
                            &basis,
                            [
                                entity.scale * 0.48,
                                entity.scale * 1.25,
                                entity.scale * 0.58,
                                entity.scale * 1.12,
                            ],
                            [
                                (entity.material.saturating_add(1)) as u32,
                                entity.material as u32,
                                (entity.material.saturating_add(2)) as u32,
                                entity.material as u32,
                                (entity.material.saturating_add(3)) as u32,
                                entity.material as u32,
                                (entity.material.saturating_add(1)) as u32,
                                entity.material as u32,
                            ],
                        ));
                    }
                    multiplayer::EntityKind::MobCreeper4d => {
                        let basis = orthonormal_basis_from_forward(entity.render_orientation);
                        instances.push(build_centered_model_instance(
                            entity.render_position,
                            &basis,
                            [
                                entity.scale * 0.92,
                                entity.scale * 1.18,
                                entity.scale * 0.92,
                                entity.scale * 1.18,
                            ],
                            [
                                (entity.material.saturating_add(5)) as u32,
                                (entity.material.saturating_add(1)) as u32,
                                (entity.material.saturating_add(3)) as u32,
                                entity.material as u32,
                                (entity.material.saturating_add(6)) as u32,
                                (entity.material.saturating_add(2)) as u32,
                                (entity.material.saturating_add(4)) as u32,
                                entity.material as u32,
                            ],
                        ));
                    }
                    multiplayer::EntityKind::MobPhaseSpider => {
                        let basis = orthonormal_basis_from_forward(entity.render_orientation);
                        let anim_t = entity.last_received_at.elapsed().as_secs_f32() * 6.0
                            + entity_id as f32 * 0.23;
                        let body_bob = 0.05 * (anim_t * 0.7).sin();
                        let body_center = offset_point_along_basis(
                            entity.render_position,
                            &basis,
                            [0.0, 0.08 + body_bob, 0.0, 0.0],
                        );
                        instances.push(build_centered_model_instance(
                            body_center,
                            &basis,
                            [
                                entity.scale * 0.88,
                                entity.scale * 0.36,
                                entity.scale * 0.78,
                                entity.scale * 0.96,
                            ],
                            [
                                (entity.material.saturating_add(2)) as u32,
                                entity.material as u32,
                                (entity.material.saturating_add(4)) as u32,
                                entity.material as u32,
                                (entity.material.saturating_add(3)) as u32,
                                entity.material as u32,
                                (entity.material.saturating_add(5)) as u32,
                                entity.material as u32,
                            ],
                        ));

                        let core_center =
                            offset_point_along_basis(body_center, &basis, [0.0, 0.11, 0.0, 0.0]);
                        instances.push(build_centered_model_instance(
                            core_center,
                            &basis,
                            [
                                entity.scale * 0.32,
                                entity.scale * 0.30,
                                entity.scale * 0.32,
                                entity.scale * 0.32,
                            ],
                            [
                                (entity.material.saturating_add(7)) as u32,
                                (entity.material.saturating_add(9)) as u32,
                                (entity.material.saturating_add(8)) as u32,
                                (entity.material.saturating_add(9)) as u32,
                                (entity.material.saturating_add(7)) as u32,
                                (entity.material.saturating_add(9)) as u32,
                                (entity.material.saturating_add(8)) as u32,
                                (entity.material.saturating_add(9)) as u32,
                            ],
                        ));

                        let splay = 0.12 * anim_t.sin();
                        let leg_y = -0.16 + 0.04 * (anim_t * 1.5).cos();
                        let leg_specs = [
                            (
                                [0.56 + splay, leg_y, 0.42, 0.20],
                                [0.72, 0.08, 0.14, 0.18],
                                1u8,
                            ),
                            (
                                [0.56 + splay, leg_y, -0.42, -0.20],
                                [0.72, 0.08, 0.14, 0.18],
                                2u8,
                            ),
                            (
                                [-0.56 - splay, leg_y, 0.42, -0.20],
                                [0.72, 0.08, 0.14, 0.18],
                                3u8,
                            ),
                            (
                                [-0.56 - splay, leg_y, -0.42, 0.20],
                                [0.72, 0.08, 0.14, 0.18],
                                4u8,
                            ),
                            (
                                [0.20, leg_y, 0.52 + splay, 0.56],
                                [0.16, 0.08, 0.72, 0.18],
                                5u8,
                            ),
                            (
                                [-0.20, leg_y, -0.52 - splay, -0.56],
                                [0.16, 0.08, 0.72, 0.18],
                                6u8,
                            ),
                            (
                                [0.20, leg_y, -0.52 - splay, 0.56],
                                [0.16, 0.08, 0.72, 0.18],
                                7u8,
                            ),
                            (
                                [-0.20, leg_y, 0.52 + splay, -0.56],
                                [0.16, 0.08, 0.72, 0.18],
                                8u8,
                            ),
                        ];
                        for (offset, axis_scale, material_bias) in leg_specs {
                            let leg_center = offset_point_along_basis(body_center, &basis, offset);
                            instances.push(build_centered_model_instance(
                                leg_center,
                                &basis,
                                [
                                    entity.scale * axis_scale[0],
                                    entity.scale * axis_scale[1],
                                    entity.scale * axis_scale[2],
                                    entity.scale * axis_scale[3],
                                ],
                                [
                                    entity.material as u32,
                                    (entity.material.saturating_add(material_bias)) as u32,
                                    entity.material as u32,
                                    (entity.material.saturating_add(material_bias)) as u32,
                                    entity.material as u32,
                                    (entity.material.saturating_add(material_bias)) as u32,
                                    entity.material as u32,
                                    (entity.material.saturating_add(material_bias)) as u32,
                                ],
                            ));
                        }
                    }
                }
            }
        }
        instances
    }

    pub(super) fn remote_player_tags(
        &self,
        view_matrix: &ndarray::Array2<f32>,
        look_dir: [f32; 4],
        focal_length_xy: f32,
        aspect: f32,
    ) -> Vec<HudPlayerTag> {
        let forward = normalize4_with_fallback(look_dir, [0.0, 0.0, 1.0, 0.0]);
        let mut ids: Vec<u64> = self.remote_players.keys().copied().collect();
        ids.sort_unstable();

        let mut tags = Vec::with_capacity(ids.len().min(REMOTE_PLAYER_TAG_MAX_COUNT));
        for entity_id in ids {
            if tags.len() >= REMOTE_PLAYER_TAG_MAX_COUNT {
                break;
            }
            let Some(player) = self.remote_players.get(&entity_id) else {
                continue;
            };

            let anchor = [
                player.render_position[0],
                player.render_position[1] + 0.48,
                player.render_position[2],
                player.render_position[3],
            ];
            let to_anchor = [
                anchor[0] - self.camera.position[0],
                anchor[1] - self.camera.position[1],
                anchor[2] - self.camera.position[2],
                anchor[3] - self.camera.position[3],
            ];
            let distance_sq = dot4(to_anchor, to_anchor);
            if !distance_sq.is_finite() || distance_sq <= 1e-6 {
                continue;
            }
            let distance = distance_sq.sqrt();
            let inv_distance = distance.recip();
            let dir = [
                to_anchor[0] * inv_distance,
                to_anchor[1] * inv_distance,
                to_anchor[2] * inv_distance,
                to_anchor[3] * inv_distance,
            ];

            if dot4(dir, forward) < REMOTE_PLAYER_TAG_FOV_DOT_MIN {
                continue;
            }

            let Some((ndc, _depth)) =
                project_world_point_to_ndc_with_depth(view_matrix, anchor, focal_length_xy, aspect)
            else {
                continue;
            };
            if ndc[0].abs() > 1.10 || ndc[1].abs() > 1.10 {
                continue;
            }

            let scale = (3.2 / (distance + 1.0)).clamp(0.55, 1.45);
            let bg_alpha = (0.82 - distance * 0.05).clamp(0.35, 0.82);
            let text = if player.name.trim().is_empty() {
                player
                    .owner_client_id
                    .map(|id| format!("player-{id}"))
                    .unwrap_or_else(|| format!("entity-{entity_id}"))
            } else {
                player.name.clone()
            };

            tags.push(HudPlayerTag {
                text,
                anchor_ndc: ndc,
                scale,
                bg_alpha,
            });
        }

        tags
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn apply_region_clock_updates_ignores_regressions() {
        let mut clocks = HashMap::new();
        clocks.insert([0, 0, 0, 0], 5);

        let regressed = apply_region_clock_updates(
            &mut clocks,
            vec![([0, 0, 0, 0], 4), ([0, 0, 0, 0], 6), ([1, 0, 0, 0], 2)],
        );

        assert_eq!(regressed, 1);
        assert_eq!(clocks.get(&[0, 0, 0, 0]), Some(&6));
        assert_eq!(clocks.get(&[1, 0, 0, 0]), Some(&2));
    }

    #[test]
    fn region_clock_precondition_mismatch_count_treats_missing_as_zero() {
        let mut clocks = HashMap::new();
        clocks.insert([0, 0, 0, 0], 3);

        let preconditions = vec![
            RegionClockPrecondition {
                region_id: RegionId {
                    x: 0,
                    y: 0,
                    z: 0,
                    w: 0,
                },
                expected_clock: 3,
            },
            RegionClockPrecondition {
                region_id: RegionId {
                    x: 1,
                    y: 0,
                    z: 0,
                    w: 0,
                },
                expected_clock: 0,
            },
            RegionClockPrecondition {
                region_id: RegionId {
                    x: 2,
                    y: 0,
                    z: 0,
                    w: 0,
                },
                expected_clock: 9,
            },
        ];

        let mismatches = region_clock_precondition_mismatch_count(&clocks, &preconditions);
        assert_eq!(mismatches, 1);
    }

    #[test]
    fn region_resync_entries_for_bounds_reads_existing_and_default_clocks() {
        let mut clocks = HashMap::new();
        clocks.insert([0, 0, 0, 0], 9);
        let bounds = Aabb4i::new([0, 0, 0, 0], [7, 3, 3, 3]);

        let entries = region_resync_entries_for_bounds(&clocks, bounds, 16);
        assert_eq!(entries.len(), 2);
        assert_eq!(
            entries[0].region_id,
            RegionId {
                x: 0,
                y: 0,
                z: 0,
                w: 0
            }
        );
        assert_eq!(entries[0].client_clock, 9);
        assert_eq!(
            entries[1].region_id,
            RegionId {
                x: 1,
                y: 0,
                z: 0,
                w: 0
            }
        );
        assert_eq!(entries[1].client_clock, 0);
    }

    #[test]
    fn region_resync_entries_for_mismatched_preconditions_filters_equal_clocks() {
        let mut clocks = HashMap::new();
        clocks.insert([0, 0, 0, 0], 3);

        let preconditions = vec![
            RegionClockPrecondition {
                region_id: RegionId {
                    x: 0,
                    y: 0,
                    z: 0,
                    w: 0,
                },
                expected_clock: 3,
            },
            RegionClockPrecondition {
                region_id: RegionId {
                    x: 1,
                    y: 0,
                    z: 0,
                    w: 0,
                },
                expected_clock: 8,
            },
        ];

        let entries =
            region_resync_entries_for_mismatched_preconditions(&clocks, &preconditions, 16);
        assert_eq!(entries.len(), 1);
        assert_eq!(
            entries[0].region_id,
            RegionId {
                x: 1,
                y: 0,
                z: 0,
                w: 0
            }
        );
        assert_eq!(entries[0].client_clock, 0);
    }
}
