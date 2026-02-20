use super::*;
use higher_dimension_playground::render::{
    OVERLAY_EDGE_TAG_PLACE, OVERLAY_EDGE_TAG_REGION_BRANCH, OVERLAY_EDGE_TAG_REGION_CHUNK_ARRAY,
    OVERLAY_EDGE_TAG_REGION_EMPTY, OVERLAY_EDGE_TAG_REGION_PROCEDURAL,
    OVERLAY_EDGE_TAG_REGION_UNIFORM, OVERLAY_EDGE_TAG_TARGET,
};
use polychora::shared::chunk_payload::{ChunkArrayData, ChunkPayload};
use polychora::shared::region_tree::{
    collect_non_empty_chunks_from_core_in_bounds, RegionNodeKind, RegionTreeCore,
};
use polychora::shared::spatial::Aabb4i;
use std::collections::{HashSet, VecDeque};
use std::hash::{Hash, Hasher};

#[derive(Copy, Clone, Debug)]
enum RegionTreeDiagKind {
    Branch,
    Empty,
    Uniform(u16),
    ChunkArray,
    ProceduralRef,
}

#[derive(Copy, Clone, Debug)]
struct RegionTreeDiagNode {
    bounds: Aabb4i,
    depth: usize,
    kind: RegionTreeDiagKind,
}

#[derive(Copy, Clone, Debug)]
struct RegionTreeDiagStyle {
    edge_tag: u32,
    short_code: &'static str,
    text_color: [f32; 4],
    border_color: [f32; 4],
    connector_color: [f32; 4],
}

#[derive(Copy, Clone, Debug)]
struct ProjectedRegionBounds {
    anchor_ndc: [f32; 2],
    rect_ndc: [f32; 4],
    depth: f32,
    area_ndc: f32,
}

fn region_tree_diag_kind(kind: &RegionNodeKind) -> RegionTreeDiagKind {
    match kind {
        RegionNodeKind::Branch(_) => RegionTreeDiagKind::Branch,
        RegionNodeKind::Empty => RegionTreeDiagKind::Empty,
        RegionNodeKind::Uniform(material) => RegionTreeDiagKind::Uniform(*material),
        RegionNodeKind::ChunkArray(_) => RegionTreeDiagKind::ChunkArray,
        RegionNodeKind::ProceduralRef(_) => RegionTreeDiagKind::ProceduralRef,
    }
}

fn region_tree_diag_style(kind: RegionTreeDiagKind) -> RegionTreeDiagStyle {
    match kind {
        RegionTreeDiagKind::Branch => RegionTreeDiagStyle {
            edge_tag: OVERLAY_EDGE_TAG_REGION_BRANCH,
            short_code: "BR",
            text_color: [1.00, 0.86, 0.56, 1.00],
            border_color: [0.96, 0.66, 0.18, 0.90],
            connector_color: [0.96, 0.66, 0.18, 0.70],
        },
        RegionTreeDiagKind::Empty => RegionTreeDiagStyle {
            edge_tag: OVERLAY_EDGE_TAG_REGION_EMPTY,
            short_code: "EM",
            text_color: [0.92, 0.93, 0.96, 0.96],
            border_color: [0.62, 0.66, 0.76, 0.72],
            connector_color: [0.62, 0.66, 0.76, 0.52],
        },
        RegionTreeDiagKind::Uniform(_) => RegionTreeDiagStyle {
            edge_tag: OVERLAY_EDGE_TAG_REGION_UNIFORM,
            short_code: "UF",
            text_color: [0.80, 0.98, 0.78, 1.00],
            border_color: [0.36, 0.84, 0.34, 0.88],
            connector_color: [0.36, 0.84, 0.34, 0.70],
        },
        RegionTreeDiagKind::ChunkArray => RegionTreeDiagStyle {
            edge_tag: OVERLAY_EDGE_TAG_REGION_CHUNK_ARRAY,
            short_code: "CA",
            text_color: [0.78, 0.96, 1.00, 1.00],
            border_color: [0.22, 0.80, 0.90, 0.90],
            connector_color: [0.22, 0.80, 0.90, 0.72],
        },
        RegionTreeDiagKind::ProceduralRef => RegionTreeDiagStyle {
            edge_tag: OVERLAY_EDGE_TAG_REGION_PROCEDURAL,
            short_code: "PR",
            text_color: [0.96, 0.84, 1.00, 1.00],
            border_color: [0.72, 0.44, 0.96, 0.88],
            connector_color: [0.72, 0.44, 0.96, 0.70],
        },
    }
}

fn region_tree_diag_label_text(node: RegionTreeDiagNode) -> String {
    let style = region_tree_diag_style(node.kind);
    let span = [
        node.bounds.max[0]
            .saturating_sub(node.bounds.min[0])
            .saturating_add(1),
        node.bounds.max[1]
            .saturating_sub(node.bounds.min[1])
            .saturating_add(1),
        node.bounds.max[2]
            .saturating_sub(node.bounds.min[2])
            .saturating_add(1),
        node.bounds.max[3]
            .saturating_sub(node.bounds.min[3])
            .saturating_add(1),
    ];
    let kind_suffix = match node.kind {
        RegionTreeDiagKind::Uniform(material) => format!(" m{material}"),
        _ => String::new(),
    };
    format!(
        "{} d{} {}x{}x{}x{}{}",
        style.short_code, node.depth, span[0], span[1], span[2], span[3], kind_suffix
    )
}

fn project_chunk_bounds_to_ndc(
    bounds: Aabb4i,
    view_matrix: &ndarray::Array2<f32>,
    focal_length_xy: f32,
    aspect: f32,
) -> Option<ProjectedRegionBounds> {
    if !bounds.is_valid() {
        return None;
    }

    let chunk_size = voxel::CHUNK_SIZE as i32;
    let min_world = [
        bounds.min[0].saturating_mul(chunk_size) as f32,
        bounds.min[1].saturating_mul(chunk_size) as f32,
        bounds.min[2].saturating_mul(chunk_size) as f32,
        bounds.min[3].saturating_mul(chunk_size) as f32,
    ];
    let max_world = [
        bounds.max[0].saturating_add(1).saturating_mul(chunk_size) as f32,
        bounds.max[1].saturating_add(1).saturating_mul(chunk_size) as f32,
        bounds.max[2].saturating_add(1).saturating_mul(chunk_size) as f32,
        bounds.max[3].saturating_add(1).saturating_mul(chunk_size) as f32,
    ];

    let mut min_ndc = [f32::INFINITY; 2];
    let mut max_ndc = [f32::NEG_INFINITY; 2];
    let mut nearest_depth = f32::INFINITY;
    let mut projected_count = 0usize;

    for mask in 0..16u32 {
        let mut corner = [0.0f32; 4];
        for axis in 0..4 {
            corner[axis] = if (mask >> axis) & 1 == 0 {
                min_world[axis]
            } else {
                max_world[axis]
            };
        }
        let Some((ndc, depth)) =
            project_world_point_to_ndc_with_depth(view_matrix, corner, focal_length_xy, aspect)
        else {
            continue;
        };
        min_ndc[0] = min_ndc[0].min(ndc[0]);
        min_ndc[1] = min_ndc[1].min(ndc[1]);
        max_ndc[0] = max_ndc[0].max(ndc[0]);
        max_ndc[1] = max_ndc[1].max(ndc[1]);
        nearest_depth = nearest_depth.min(depth);
        projected_count = projected_count.saturating_add(1);
    }

    if projected_count == 0 {
        return None;
    }

    let center_world = [
        0.5 * (min_world[0] + max_world[0]),
        0.5 * (min_world[1] + max_world[1]),
        0.5 * (min_world[2] + max_world[2]),
        0.5 * (min_world[3] + max_world[3]),
    ];
    let anchor_ndc =
        project_world_point_to_ndc_with_depth(view_matrix, center_world, focal_length_xy, aspect)
            .map(|(ndc, _)| ndc)
            .unwrap_or([
                0.5 * (min_ndc[0] + max_ndc[0]),
                0.5 * (min_ndc[1] + max_ndc[1]),
            ]);

    if max_ndc[0] < -1.35 || min_ndc[0] > 1.35 || max_ndc[1] < -1.35 || min_ndc[1] > 1.35 {
        return None;
    }

    let width = (max_ndc[0] - min_ndc[0]).max(0.0);
    let height = (max_ndc[1] - min_ndc[1]).max(0.0);
    Some(ProjectedRegionBounds {
        anchor_ndc,
        rect_ndc: [min_ndc[0], min_ndc[1], max_ndc[0], max_ndc[1]],
        depth: nearest_depth,
        area_ndc: width * height,
    })
}

fn collect_region_tree_bounds_hierarchy(
    node: &RegionTreeCore,
    max_nodes: usize,
    non_empty_only: bool,
    out: &mut Vec<RegionTreeDiagNode>,
) {
    if max_nodes == 0 || out.len() >= max_nodes {
        return;
    }

    // Breadth-first sampling avoids DFS bias when max_nodes truncates the hierarchy.
    let mut queue: VecDeque<(&RegionTreeCore, usize)> = VecDeque::new();
    queue.push_back((node, 0));
    while let Some((current, depth)) = queue.pop_front() {
        if non_empty_only && !region_tree_node_has_non_empty_content(current) {
            continue;
        }
        out.push(RegionTreeDiagNode {
            bounds: current.bounds,
            depth,
            kind: region_tree_diag_kind(&current.kind),
        });
        if out.len() >= max_nodes {
            return;
        }

        if let RegionNodeKind::Branch(children) = &current.kind {
            for child in children {
                queue.push_back((child, depth.saturating_add(1)));
            }
        }
    }
}

fn chunk_payload_has_non_empty_material(payload: &ChunkPayload) -> bool {
    match payload {
        ChunkPayload::Empty => false,
        ChunkPayload::Uniform(material) => *material != 0,
        ChunkPayload::Dense16 { materials } => materials.iter().any(|material| *material != 0),
        ChunkPayload::PalettePacked { .. } => payload
            .dense_materials()
            .map(|materials| materials.into_iter().any(|material| material != 0))
            .unwrap_or(true),
    }
}

fn chunk_array_has_non_empty_content(chunk_array: &ChunkArrayData) -> bool {
    if let Some(default_chunk_idx) = chunk_array.default_chunk_idx {
        if let Some(default_payload) = chunk_array.chunk_palette.get(default_chunk_idx as usize) {
            if chunk_payload_has_non_empty_material(default_payload) {
                return true;
            }
        } else {
            // Conservatively keep malformed payloads visible in diagnostics.
            return true;
        }
    }

    let Ok(indices) = chunk_array.decode_dense_indices() else {
        return true;
    };
    indices.into_iter().any(|palette_idx| {
        chunk_array
            .chunk_palette
            .get(palette_idx as usize)
            .map(chunk_payload_has_non_empty_material)
            .unwrap_or(true)
    })
}

fn region_tree_node_has_non_empty_content(node: &RegionTreeCore) -> bool {
    match &node.kind {
        RegionNodeKind::Empty | RegionNodeKind::ProceduralRef(_) => false,
        RegionNodeKind::Uniform(material) => *material != 0,
        RegionNodeKind::ChunkArray(chunk_array) => chunk_array_has_non_empty_content(chunk_array),
        RegionNodeKind::Branch(children) => {
            children.iter().any(region_tree_node_has_non_empty_content)
        }
    }
}

#[derive(Default)]
struct RegionTreeNodeStats {
    node_count: usize,
    branch_count: usize,
    empty_count: usize,
    uniform_count: usize,
    chunk_array_count: usize,
    procedural_ref_count: usize,
    max_depth: usize,
}

fn collect_region_tree_node_stats(node: &RegionTreeCore) -> RegionTreeNodeStats {
    fn recurse(node: &RegionTreeCore, depth: usize, stats: &mut RegionTreeNodeStats) {
        stats.node_count = stats.node_count.saturating_add(1);
        stats.max_depth = stats.max_depth.max(depth);
        match &node.kind {
            RegionNodeKind::Empty => {
                stats.empty_count = stats.empty_count.saturating_add(1);
            }
            RegionNodeKind::Uniform(_) => {
                stats.uniform_count = stats.uniform_count.saturating_add(1);
            }
            RegionNodeKind::ChunkArray(_) => {
                stats.chunk_array_count = stats.chunk_array_count.saturating_add(1);
            }
            RegionNodeKind::ProceduralRef(_) => {
                stats.procedural_ref_count = stats.procedural_ref_count.saturating_add(1);
            }
            RegionNodeKind::Branch(children) => {
                stats.branch_count = stats.branch_count.saturating_add(1);
                for child in children {
                    recurse(child, depth.saturating_add(1), stats);
                }
            }
        }
    }

    let mut stats = RegionTreeNodeStats::default();
    recurse(node, 0, &mut stats);
    stats
}

fn tight_bounds_from_chunk_positions(
    positions: impl IntoIterator<Item = [i32; 4]>,
) -> Option<Aabb4i> {
    let mut iter = positions.into_iter();
    let first = iter.next()?;
    let mut min = first;
    let mut max = first;
    for pos in iter {
        for axis in 0..4 {
            min[axis] = min[axis].min(pos[axis]);
            max[axis] = max[axis].max(pos[axis]);
        }
    }
    Some(Aabb4i::new(min, max))
}

fn world_chunk_from_position(position: [f32; 4]) -> [i32; 4] {
    let chunk_size = voxel::CHUNK_SIZE as i32;
    [
        (position[0].floor() as i32).div_euclid(chunk_size),
        (position[1].floor() as i32).div_euclid(chunk_size),
        (position[2].floor() as i32).div_euclid(chunk_size),
        (position[3].floor() as i32).div_euclid(chunk_size),
    ]
}

fn world_request_bounds_for_center_chunk(center_chunk: [i32; 4], radius_chunks: i32) -> Aabb4i {
    let radius = radius_chunks.max(0);
    Aabb4i::new(
        [
            center_chunk[0].saturating_sub(radius),
            center_chunk[1].saturating_sub(radius),
            center_chunk[2].saturating_sub(radius),
            center_chunk[3].saturating_sub(radius),
        ],
        [
            center_chunk[0].saturating_add(radius),
            center_chunk[1].saturating_add(radius),
            center_chunk[2].saturating_add(radius),
            center_chunk[3].saturating_add(radius),
        ],
    )
}

fn world_request_radius_chunks_for_distance(distance_world: f32) -> i32 {
    let chunk_size = voxel::CHUNK_SIZE as f32;
    // One-chunk margin prevents request thrash at the interest boundary.
    ((distance_world.max(0.0) / chunk_size).ceil() as i32)
        .saturating_add(1)
        .max(1)
}

fn zero_dense_chunk_materials() -> Vec<u16> {
    vec![0u16; polychora::shared::voxel::CHUNK_VOLUME]
}

fn dense_materials_from_payload_or_zero(payload: Option<ChunkPayload>) -> Vec<u16> {
    let Some(payload) = payload else {
        return zero_dense_chunk_materials();
    };
    let Ok(materials) = payload.dense_materials() else {
        return zero_dense_chunk_materials();
    };
    if materials.len() == polychora::shared::voxel::CHUNK_VOLUME {
        materials
    } else {
        zero_dense_chunk_materials()
    }
}

fn dense_materials_from_region_core_chunk(core: &RegionTreeCore, chunk: [i32; 4]) -> Vec<u16> {
    let bounds = Aabb4i::new(chunk, chunk);
    let chunks = collect_non_empty_chunks_from_core_in_bounds(core, bounds);
    for (key, payload) in chunks {
        if key == chunk {
            return dense_materials_from_payload_or_zero(Some(payload));
        }
    }
    zero_dense_chunk_materials()
}

fn dense_materials_hash(materials: &[u16]) -> u64 {
    let mut hasher = std::collections::hash_map::DefaultHasher::new();
    materials.hash(&mut hasher);
    hasher.finish()
}

fn dense_materials_non_zero_count(materials: &[u16]) -> usize {
    materials.iter().filter(|m| **m != 0).count()
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

    fn next_multiplayer_chunk_sample_diag_u32(&mut self) -> u32 {
        self.multiplayer_chunk_sample_diag_rng_state = self
            .multiplayer_chunk_sample_diag_rng_state
            .wrapping_mul(6364136223846793005)
            .wrapping_add(1442695040888963407);
        (self.multiplayer_chunk_sample_diag_rng_state >> 32) as u32
    }

    fn next_multiplayer_chunk_sample_diag_chunk_in_bounds(&mut self, bounds: Aabb4i) -> [i32; 4] {
        let mut chunk = [0i32; 4];
        for axis in 0..4 {
            let lo = bounds.min[axis];
            let hi = bounds.max[axis];
            if hi <= lo {
                chunk[axis] = lo;
                continue;
            }
            let span = (hi - lo + 1) as u32;
            chunk[axis] = lo + (self.next_multiplayer_chunk_sample_diag_u32() % span) as i32;
        }
        chunk
    }

    pub(super) fn send_multiplayer_chunk_sample_diag_request(&mut self) {
        if !self.multiplayer_chunk_sample_diag_enabled {
            return;
        }
        let Some(bounds) = self.multiplayer_last_world_request_bounds else {
            return;
        };
        if !bounds.is_valid() {
            return;
        }
        let chunk = self.next_multiplayer_chunk_sample_diag_chunk_in_bounds(bounds);
        let request_id = self.multiplayer_chunk_sample_diag_next_request_id;
        self.multiplayer_chunk_sample_diag_next_request_id = self
            .multiplayer_chunk_sample_diag_next_request_id
            .wrapping_add(1)
            .max(1);

        let Some(client) = self.multiplayer.as_ref() else {
            return;
        };
        client.send(MultiplayerClientMessage::WorldChunkSampleRequest { request_id, chunk });
    }

    fn record_multiplayer_chunk_sample_diag_patch(&mut self, patch: &RegionTreeCore) {
        if !self.multiplayer_chunk_sample_diag_enabled {
            return;
        }
        self.multiplayer_chunk_sample_diag_patch_seq = self
            .multiplayer_chunk_sample_diag_patch_seq
            .wrapping_add(1)
            .max(1);
        self.multiplayer_chunk_sample_diag_recent_patches
            .push_back((self.multiplayer_chunk_sample_diag_patch_seq, patch.clone()));
        while self.multiplayer_chunk_sample_diag_recent_patches.len()
            > self.multiplayer_chunk_sample_diag_history_limit
        {
            self.multiplayer_chunk_sample_diag_recent_patches
                .pop_front();
        }
    }

    fn handle_multiplayer_chunk_sample_diag_response(
        &mut self,
        request_id: u64,
        chunk: [i32; 4],
        dense_materials: Vec<u16>,
    ) {
        if !self.multiplayer_chunk_sample_diag_enabled {
            return;
        }
        if dense_materials.len() != polychora::shared::voxel::CHUNK_VOLUME {
            eprintln!(
                "[client-chunk-sample-diag] invalid dense response: request_id={} chunk={:?} len={} expected={}",
                request_id,
                chunk,
                dense_materials.len(),
                polychora::shared::voxel::CHUNK_VOLUME
            );
            return;
        }

        let world_dense =
            dense_materials_from_payload_or_zero(self.scene.debug_world_tree_chunk_payload(chunk));
        let render_bounds = self
            .multiplayer_last_world_request_bounds
            .unwrap_or_else(|| Aabb4i::new(chunk, chunk));
        let render_payloads = self
            .scene
            .debug_render_bvh_chunk_payloads_in_bounds(render_bounds, chunk);
        let render_dense_variants: Vec<Vec<u16>> = render_payloads
            .iter()
            .cloned()
            .map(|payload| dense_materials_from_payload_or_zero(Some(payload)))
            .collect();
        let render_dense = render_dense_variants
            .first()
            .cloned()
            .unwrap_or_else(zero_dense_chunk_materials);
        let render_conflict = render_dense_variants
            .iter()
            .skip(1)
            .any(|dense| *dense != render_dense);

        let world_match = world_dense == dense_materials;
        let render_match = render_dense == dense_materials;

        let mut overlap_count = 0usize;
        let mut overlap_mismatch_count = 0usize;
        let mut newest_overlap_seq = None;
        let mut newest_overlap_match = None;
        for (seq, patch) in self
            .multiplayer_chunk_sample_diag_recent_patches
            .iter()
            .rev()
        {
            if !patch.bounds.contains_chunk(chunk) {
                continue;
            }
            let patch_dense = dense_materials_from_region_core_chunk(patch, chunk);
            let patch_match = patch_dense == dense_materials;
            if newest_overlap_seq.is_none() {
                newest_overlap_seq = Some(*seq);
                newest_overlap_match = Some(patch_match);
            }
            overlap_count = overlap_count.saturating_add(1);
            if !patch_match {
                overlap_mismatch_count = overlap_mismatch_count.saturating_add(1);
            }
        }

        if !world_match || !render_match || render_conflict || newest_overlap_match == Some(false) {
            eprintln!(
                "[client-chunk-sample-diag] request_id={} chunk={:?} server(hash={},nz={}) world(match={},hash={},nz={}) render(match={},hash={},nz={},payloads={},conflict={}) patch_overlaps={} patch_mismatches={} newest_patch_seq={:?} newest_patch_match={:?}",
                request_id,
                chunk,
                dense_materials_hash(&dense_materials),
                dense_materials_non_zero_count(&dense_materials),
                world_match,
                dense_materials_hash(&world_dense),
                dense_materials_non_zero_count(&world_dense),
                render_match,
                dense_materials_hash(&render_dense),
                dense_materials_non_zero_count(&render_dense),
                render_payloads.len(),
                render_conflict,
                overlap_count,
                overlap_mismatch_count,
                newest_overlap_seq,
                newest_overlap_match
            );
        }
    }

    fn send_multiplayer_world_interest_update(&mut self, bounds: Aabb4i, reason: &str) {
        if !bounds.is_valid() {
            return;
        }
        if let Some(client) = self.multiplayer.as_ref() {
            client.send(MultiplayerClientMessage::WorldInterestUpdate { bounds });
            eprintln!(
                "Sent multiplayer world interest update: reason={reason} bounds={:?}->{:?}",
                bounds.min, bounds.max
            );
        }
    }

    fn maybe_request_multiplayer_world_for_position(&mut self, position: [f32; 4], reason: &str) {
        let center_chunk = world_chunk_from_position(position);
        let request_radius_chunks =
            world_request_radius_chunks_for_distance(self.vte_max_trace_distance);
        let desired_bounds =
            world_request_bounds_for_center_chunk(center_chunk, request_radius_chunks);
        if self.multiplayer_last_world_request_bounds == Some(desired_bounds) {
            return;
        }
        self.send_multiplayer_world_interest_update(desired_bounds, reason);
        self.multiplayer_last_world_request_center_chunk = Some(center_chunk);
        self.multiplayer_last_world_request_bounds = Some(desired_bounds);
    }

    pub(super) fn append_multiplayer_stream_tree_diag_overlay_instances(
        &self,
        overlay_edge_instances: &mut Vec<common::ModelInstance>,
    ) {
        if !self.multiplayer_stream_tree_diag_enabled {
            return;
        }
        let Some(root) = self.multiplayer_stream_tree_diag.root() else {
            return;
        };

        let mut bounds_hierarchy = Vec::new();
        collect_region_tree_bounds_hierarchy(
            root,
            self.multiplayer_stream_tree_diag_max_nodes.max(1),
            self.multiplayer_stream_tree_diag_non_empty_only,
            &mut bounds_hierarchy,
        );
        for node in bounds_hierarchy {
            let style = region_tree_diag_style(node.kind);
            append_chunk_bounds_outline_edge_instance(
                overlay_edge_instances,
                node.bounds.min,
                node.bounds.max,
                style.edge_tag,
            );
        }
    }

    pub(super) fn append_multiplayer_stream_tree_diag_hud_tags(
        &self,
        hud_tags: &mut Vec<HudPlayerTag>,
        view_matrix: &ndarray::Array2<f32>,
        focal_length_xy: f32,
        aspect: f32,
    ) {
        if !self.multiplayer_stream_tree_diag_enabled
            || !self.multiplayer_stream_tree_diag_labels_enabled
        {
            return;
        }
        let Some(root) = self.multiplayer_stream_tree_diag.root() else {
            return;
        };

        let mut nodes = Vec::new();
        collect_region_tree_bounds_hierarchy(
            root,
            self.multiplayer_stream_tree_diag_max_nodes.max(1),
            self.multiplayer_stream_tree_diag_non_empty_only,
            &mut nodes,
        );

        let mut projected = Vec::<(RegionTreeDiagNode, ProjectedRegionBounds)>::new();
        projected.reserve(nodes.len());
        for node in nodes {
            let Some(projection) =
                project_chunk_bounds_to_ndc(node.bounds, view_matrix, focal_length_xy, aspect)
            else {
                continue;
            };
            projected.push((node, projection));
        }

        projected.sort_by(|left, right| {
            right
                .1
                .area_ndc
                .partial_cmp(&left.1.area_ndc)
                .unwrap_or(std::cmp::Ordering::Equal)
                .then_with(|| {
                    left.1
                        .depth
                        .partial_cmp(&right.1.depth)
                        .unwrap_or(std::cmp::Ordering::Equal)
                })
                .then_with(|| left.0.depth.cmp(&right.0.depth))
        });

        for (node, projection) in projected
            .into_iter()
            .take(self.multiplayer_stream_tree_diag_max_labels.max(1))
        {
            let style = region_tree_diag_style(node.kind);
            let scale = (1.30 - (node.depth as f32) * 0.06).clamp(0.58, 1.28);
            hud_tags.push(HudPlayerTag {
                text: region_tree_diag_label_text(node),
                anchor_ndc: projection.anchor_ndc,
                scale,
                bg_alpha: 0.58,
                text_color: style.text_color,
                border_color: style.border_color,
                connector_color: style.connector_color,
                target_rect_ndc: Some(projection.rect_ndc),
            });
        }
    }

    pub(super) fn append_multiplayer_stream_tree_compare_overlay_instances(
        &mut self,
        overlay_edge_instances: &mut Vec<common::ModelInstance>,
    ) {
        if !self.multiplayer_stream_tree_compare_diag_enabled {
            return;
        }

        self.multiplayer_stream_tree_compare_diag_frame_counter = self
            .multiplayer_stream_tree_compare_diag_frame_counter
            .saturating_add(1);
        let sample_cap = self.multiplayer_stream_tree_compare_diag_max_chunks.max(1);

        let mut world_positions = self.scene.collect_non_empty_explicit_chunk_positions();
        world_positions.sort_unstable();
        let world_set: HashSet<[i32; 4]> = world_positions.iter().copied().collect();

        let mut tree_positions: Vec<[i32; 4]> = self
            .multiplayer_stream_tree_diag
            .collect_chunks()
            .into_iter()
            .map(|(key, _)| key)
            .collect();
        tree_positions.sort_unstable();
        let tree_set: HashSet<[i32; 4]> = tree_positions.iter().copied().collect();

        let missing_count = world_set
            .iter()
            .filter(|pos| !tree_set.contains(*pos))
            .count();
        let extra_count = tree_set
            .iter()
            .filter(|pos| !world_set.contains(*pos))
            .count();

        let mut missing_samples = Vec::new();
        for pos in world_positions.iter().copied() {
            if tree_set.contains(&pos) {
                continue;
            }
            missing_samples.push(pos);
            if missing_samples.len() >= sample_cap {
                break;
            }
        }

        let mut extra_samples = Vec::new();
        for pos in tree_positions.iter().copied() {
            if world_set.contains(&pos) {
                continue;
            }
            extra_samples.push(pos);
            if extra_samples.len() >= sample_cap {
                break;
            }
        }

        for pos in missing_samples.iter().copied() {
            append_chunk_bounds_outline_edge_instance(
                overlay_edge_instances,
                pos,
                pos,
                OVERLAY_EDGE_TAG_TARGET,
            );
        }
        for pos in extra_samples.iter().copied() {
            append_chunk_bounds_outline_edge_instance(
                overlay_edge_instances,
                pos,
                pos,
                OVERLAY_EDGE_TAG_PLACE,
            );
        }

        let world_bounds = tight_bounds_from_chunk_positions(world_positions.iter().copied());
        let tree_bounds = tight_bounds_from_chunk_positions(tree_positions.iter().copied());
        let root_bounds = self
            .multiplayer_stream_tree_diag
            .root()
            .map(|root| (root.bounds.min, root.bounds.max));
        let node_stats = self
            .multiplayer_stream_tree_diag
            .root()
            .map(collect_region_tree_node_stats)
            .unwrap_or_default();

        let mut hasher = std::collections::hash_map::DefaultHasher::new();
        self.multiplayer_stream_tree_diag_enabled.hash(&mut hasher);
        self.multiplayer_stream_tree_compare_diag_enabled
            .hash(&mut hasher);
        world_positions.len().hash(&mut hasher);
        tree_positions.len().hash(&mut hasher);
        missing_count.hash(&mut hasher);
        extra_count.hash(&mut hasher);
        for pos in missing_samples.iter().take(8) {
            pos.hash(&mut hasher);
        }
        for pos in extra_samples.iter().take(8) {
            pos.hash(&mut hasher);
        }
        world_bounds.hash(&mut hasher);
        tree_bounds.hash(&mut hasher);
        root_bounds.hash(&mut hasher);
        node_stats.node_count.hash(&mut hasher);
        node_stats.max_depth.hash(&mut hasher);
        let summary_hash = hasher.finish();

        let interval = self
            .multiplayer_stream_tree_compare_diag_log_interval
            .max(1) as u64;
        let interval_due = self.multiplayer_stream_tree_compare_diag_frame_counter % interval == 0;
        if self.multiplayer_stream_tree_compare_diag_last_hash != Some(summary_hash) || interval_due
        {
            eprintln!(
                "[client-region-tree-compare] frame={} world_chunks={} tree_chunks={} missing={} extra={} world_bounds={:?} tree_bounds={:?} root_bounds={:?} nodes={} branches={} uniform={} chunk_array={} empty={} procedural={} max_depth={} overlays(missing={},extra={})",
                self.multiplayer_stream_tree_compare_diag_frame_counter,
                world_positions.len(),
                tree_positions.len(),
                missing_count,
                extra_count,
                world_bounds,
                tree_bounds,
                root_bounds,
                node_stats.node_count,
                node_stats.branch_count,
                node_stats.uniform_count,
                node_stats.chunk_array_count,
                node_stats.empty_count,
                node_stats.procedural_ref_count,
                node_stats.max_depth,
                missing_samples.len(),
                extra_samples.len(),
            );
            if !missing_samples.is_empty() {
                eprintln!(
                    "[client-region-tree-compare] missing_samples={:?}",
                    missing_samples
                );
            }
            if !extra_samples.is_empty() {
                eprintln!(
                    "[client-region-tree-compare] extra_samples={:?}",
                    extra_samples
                );
            }
            self.multiplayer_stream_tree_compare_diag_last_hash = Some(summary_hash);
        }
    }

    pub(super) fn apply_multiplayer_world_patch(
        &mut self,
        patch: RegionTreeCore,
    ) -> Option<(usize, usize)> {
        let patch_bounds = patch.bounds;
        let patch_apply_start = Instant::now();

        let scene_patch_start = Instant::now();
        let scene_patch_stats = self.scene.apply_region_patch(patch_bounds, &patch);
        let scene_patch_elapsed_ms = scene_patch_start.elapsed().as_secs_f64() * 1000.0;
        let diag_splice_elapsed_ms = if (self.multiplayer_stream_tree_diag_enabled
            || self.multiplayer_stream_tree_compare_diag_enabled
            || self.multiplayer_stream_tree_diag_labels_enabled)
            && scene_patch_stats.changed_chunks > 0
        {
            let diag_splice_start = Instant::now();
            let _ = self
                .multiplayer_stream_tree_diag
                .splice_non_empty_core_in_bounds(patch_bounds, &patch);
            diag_splice_start.elapsed().as_secs_f64() * 1000.0
        } else {
            0.0
        };
        let patch_apply_elapsed_ms = patch_apply_start.elapsed().as_secs_f64() * 1000.0;
        if patch_apply_elapsed_ms >= 50.0 {
            eprintln!(
                "[client-world-patch] elapsed_ms={:.3} bounds={:?}->{:?} patch_cells={} previous_non_empty={} desired_non_empty={} upserts={} removals={}",
                patch_apply_elapsed_ms,
                patch_bounds.min,
                patch_bounds.max,
                patch_bounds.chunk_cell_count().unwrap_or(0),
                scene_patch_stats.previous_non_empty,
                scene_patch_stats.desired_non_empty,
                scene_patch_stats.upserts,
                scene_patch_stats.removals
            );
            eprintln!(
                "[client-world-patch-breakdown] scene_apply_ms={:.3} scene_collect_previous_ms={:.3} scene_splice_ms={:.3} scene_collect_desired_ms={:.3} scene_diff_ms={:.3} diag_splice_ms={:.3} previous_total={} desired_total={} changed_chunks={} invalidated_cached_chunks={} queued_updates={}",
                scene_patch_elapsed_ms,
                scene_patch_stats.collect_previous_ms,
                scene_patch_stats.splice_ms,
                scene_patch_stats.collect_desired_ms,
                scene_patch_stats.diff_ms,
                diag_splice_elapsed_ms,
                scene_patch_stats.previous_total_chunks,
                scene_patch_stats.desired_total_chunks,
                scene_patch_stats.changed_chunks,
                scene_patch_stats.invalidated_cached_chunks,
                scene_patch_stats.queued_updates
            );
        }
        eprintln!(
            "Applied multiplayer world patch bounds={:?}->{:?} upserts={} removals={}",
            patch_bounds.min,
            patch_bounds.max,
            scene_patch_stats.upserts,
            scene_patch_stats.removals
        );
        Some((scene_patch_stats.upserts, scene_patch_stats.removals))
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
                self.multiplayer_last_world_request_center_chunk = None;
                self.multiplayer_last_world_request_bounds = None;
                self.multiplayer_stream_tree_diag =
                    polychora::shared::region_tree::RegionChunkTree::new();
                self.multiplayer_chunk_sample_diag_recent_patches.clear();
                self.multiplayer_chunk_sample_diag_patch_seq = 0;
                self.multiplayer_chunk_sample_diag_next_request_id = 1;
                self.pending_player_movement_modifiers.clear();
                self.player_modifier_external_velocity = [0.0; 4];
                self.maybe_request_multiplayer_world_for_position(self.camera.position, "welcome");
                eprintln!(
                    "Multiplayer connected: client_id={} chunks={} server_tick_hz={:.2}",
                    client_id, world.non_empty_chunks, tick_hz
                );
            }
            multiplayer::ServerMessage::Error { message } => {
                eprintln!("Multiplayer server error: {message}");
                self.append_dev_console_log_line(format!("[server] {message}"));
            }
            multiplayer::ServerMessage::WorldSubtreePatch { subtree } => {
                self.record_multiplayer_chunk_sample_diag_patch(&subtree);
                if self.apply_multiplayer_world_patch(subtree).is_some() && !self.world_ready {
                    self.world_ready = true;
                    eprintln!("World ready: subtree patch applied");
                }
            }
            multiplayer::ServerMessage::WorldChunkSampleResponse {
                request_id,
                chunk,
                dense_materials,
            } => {
                self.handle_multiplayer_chunk_sample_diag_response(
                    request_id,
                    chunk,
                    dense_materials,
                );
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
        self.maybe_request_multiplayer_world_for_position(self.camera.position, "player_update");
    }

    pub(super) fn send_multiplayer_voxel_update(
        &mut self,
        _now: Instant,
        position: [i32; 4],
        material: u8,
    ) {
        if self.multiplayer.is_none() {
            return;
        }

        if let Some(client) = self.multiplayer.as_ref() {
            client.send(MultiplayerClientMessage::SetVoxel { position, material });
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
                text_color: [0.96, 0.97, 1.00, 1.0],
                border_color: [0.0, 0.0, 0.0, 0.0],
                connector_color: [0.0, 0.0, 0.0, 0.0],
                target_rect_ndc: None,
            });
        }

        tags
    }
}
