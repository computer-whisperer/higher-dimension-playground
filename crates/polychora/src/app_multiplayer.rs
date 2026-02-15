use super::*;

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

    pub(super) fn upsert_remote_player_snapshot(
        &mut self,
        player: multiplayer::PlayerSnapshot,
        received_at: Instant,
    ) {
        let normalized_look = normalize4_with_fallback(player.look, [0.0, 0.0, 1.0, 0.0]);
        if let Some(existing) = self.remote_players.get_mut(&player.client_id) {
            let previous_position = existing.position;
            let previous_update_ms = existing.last_update_ms;

            existing.name = player.name;
            existing.position = player.position;
            existing.look = normalized_look;
            existing.last_update_ms = player.last_update_ms;
            existing.last_received_at = received_at;

            let update_delta_ms = player.last_update_ms.saturating_sub(previous_update_ms);
            if update_delta_ms > 0 {
                let dt_s = (update_delta_ms as f32) * 0.001;
                if dt_s > 1e-4 {
                    let mut velocity = [0.0f32; 4];
                    for axis in 0..4 {
                        velocity[axis] = (player.position[axis] - previous_position[axis]) / dt_s;
                    }
                    let speed = dot4(velocity, velocity).sqrt();
                    if speed.is_finite() && speed > REMOTE_PLAYER_MAX_PREDICTED_SPEED {
                        let clamp = REMOTE_PLAYER_MAX_PREDICTED_SPEED / speed;
                        for v in &mut velocity {
                            *v *= clamp;
                        }
                    }
                    existing.velocity = velocity;
                }
            } else {
                for v in &mut existing.velocity {
                    *v *= 0.85;
                }
            }

            if distance4(previous_position, player.position) > REMOTE_PLAYER_TELEPORT_SNAP_DISTANCE
            {
                existing.render_position = existing.position;
                existing.render_look = existing.look;
                existing.velocity = [0.0; 4];
                existing.footstep_distance_accum = 0.0;
            }
            return;
        }

        self.remote_players.insert(
            player.client_id,
            RemotePlayerState {
                name: player.name,
                position: player.position,
                look: normalized_look,
                last_update_ms: player.last_update_ms,
                render_position: player.position,
                render_look: normalized_look,
                velocity: [0.0; 4],
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
            if distance4(entity.render_position, entity.position)
                > REMOTE_ENTITY_TELEPORT_SNAP_DISTANCE
            {
                entity.render_position = entity.position;
                entity.render_orientation = entity.orientation;
            } else {
                entity.render_position = lerp4(entity.render_position, entity.position, pos_alpha);
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
                self.next_multiplayer_edit_id = 1;
                self.pending_voxel_edits.clear();
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
            }
            multiplayer::ServerMessage::PlayerJoined { player } => {
                if Some(player.client_id) == self.multiplayer_self_id {
                    return;
                }
                self.upsert_remote_player_snapshot(player, received_at);
            }
            multiplayer::ServerMessage::PlayerLeft { client_id } => {
                self.remote_players.remove(&client_id);
            }
            multiplayer::ServerMessage::PlayerPositions { players, .. } => {
                let mut seen = HashSet::with_capacity(players.len());
                for player in players {
                    if Some(player.client_id) == self.multiplayer_self_id {
                        continue;
                    }
                    seen.insert(player.client_id);
                    self.upsert_remote_player_snapshot(player, received_at);
                }
                self.remote_players
                    .retain(|client_id, _| seen.contains(client_id));
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
            multiplayer::ServerMessage::Pong { .. } => {}
            multiplayer::ServerMessage::EntitySpawned { entity } => {
                self.remote_entities.insert(
                    entity.entity_id,
                    RemoteEntityState {
                        kind: entity.kind,
                        position: entity.position,
                        orientation: entity.orientation,
                        scale: entity.scale,
                        material: entity.material,
                        render_position: entity.position,
                        render_orientation: entity.orientation,
                        last_received_at: received_at,
                    },
                );
            }
            multiplayer::ServerMessage::EntityDestroyed { entity_id } => {
                self.remote_entities.remove(&entity_id);
            }
            multiplayer::ServerMessage::EntityPositions { entities, .. } => {
                let mut seen = HashSet::with_capacity(entities.len());
                for entity in entities {
                    seen.insert(entity.entity_id);
                    if let Some(existing) = self.remote_entities.get_mut(&entity.entity_id) {
                        existing.position = entity.position;
                        existing.orientation = entity.orientation;
                        existing.scale = entity.scale;
                        existing.material = entity.material;
                        existing.last_received_at = received_at;
                    } else {
                        self.remote_entities.insert(
                            entity.entity_id,
                            RemoteEntityState {
                                kind: entity.kind,
                                position: entity.position,
                                orientation: entity.orientation,
                                scale: entity.scale,
                                material: entity.material,
                                render_position: entity.position,
                                render_orientation: entity.orientation,
                                last_received_at: received_at,
                            },
                        );
                    }
                }
                self.remote_entities
                    .retain(|entity_id, _| seen.contains(entity_id));
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

    pub(super) fn remote_player_instances(&self, time_s: f32) -> Vec<common::ModelInstance> {
        let mut ids: Vec<u64> = self.remote_players.keys().copied().collect();
        ids.sort_unstable();
        let mut instances = Vec::with_capacity(ids.len() * REMOTE_AVATAR_PART_COUNT_ESTIMATE);
        for client_id in ids {
            if let Some(player) = self.remote_players.get(&client_id) {
                instances.extend(build_remote_player_avatar_instances(
                    client_id,
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
        let mut instances = Vec::with_capacity(ids.len());
        for entity_id in ids {
            if let Some(entity) = self.remote_entities.get(&entity_id) {
                match entity.kind {
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
        for client_id in ids {
            if tags.len() >= REMOTE_PLAYER_TAG_MAX_COUNT {
                break;
            }
            let Some(player) = self.remote_players.get(&client_id) else {
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
                format!("player-{client_id}")
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
