use super::*;

pub(super) fn sanitize_player_name(name: &str, client_id: u64) -> String {
    let trimmed = name.trim();
    if trimmed.is_empty() {
        return format!("player-{client_id}");
    }
    trimmed.chars().take(32).collect()
}

pub(super) fn env_flag_enabled(name: &str) -> bool {
    std::env::var(name)
        .ok()
        .map(|value| {
            let normalized = value.trim().to_ascii_lowercase();
            matches!(normalized.as_str(), "1" | "true" | "yes" | "on")
        })
        .unwrap_or(false)
}

pub(super) fn parse_spawn_vec4(args: &[&str]) -> Option<[f32; 4]> {
    if args.len() != 4 {
        return None;
    }
    let x = args[0].parse::<f32>().ok()?;
    let y = args[1].parse::<f32>().ok()?;
    let z = args[2].parse::<f32>().ok()?;
    let w = args[3].parse::<f32>().ok()?;
    Some([x, y, z, w])
}

pub(super) fn spawn_usage_string() -> String {
    let names = entity_types::spawnable_names().join("|");
    format!("Usage: /spawn <{names}> [x y z w]")
}

pub(super) fn entity_type_entry_for_token(
    token: &str,
) -> Option<&'static entity_types::EntityTypeEntry> {
    entity_types::lookup_by_name(token).filter(|entry| entry.is_spawnable())
}

pub(super) fn default_spawn_pose_for_client(
    state: &ServerState,
    client_id: u64,
) -> ([f32; 4], [f32; 4]) {
    let Some(player) = state.players.get(&client_id) else {
        return ([0.0, 0.0, 0.0, 0.0], [0.0, 0.0, 1.0, 0.0]);
    };
    let Some(snapshot) = state.entity_store.snapshot(player.entity_id) else {
        return ([0.0, 0.0, 0.0, 0.0], [0.0, 0.0, 1.0, 0.0]);
    };
    let look = normalize4_or_default(snapshot.entity.pose.orientation, [0.0, 0.0, 1.0, 0.0]);
    let position = [
        snapshot.entity.pose.position[0] + look[0] * 3.0,
        snapshot.entity.pose.position[1] + look[1] * 3.0,
        snapshot.entity.pose.position[2] + look[2] * 3.0,
        snapshot.entity.pose.position[3] + look[3] * 3.0,
    ];
    (position, look)
}

pub(super) fn phase_spider_next_phase_deadline(
    now_ms: u64,
    phase_offset: f32,
    phase_ticks: u32,
) -> u64 {
    let span = PHASE_SPIDER_PHASE_MAX_INTERVAL_MS
        .saturating_sub(PHASE_SPIDER_PHASE_MIN_INTERVAL_MS)
        .max(1);
    let wobble =
        ((phase_offset * 31.0 + phase_ticks as f32 * 1.73).sin() * 0.5 + 0.5).clamp(0.0, 1.0);
    now_ms.saturating_add(
        PHASE_SPIDER_PHASE_MIN_INTERVAL_MS + ((span as f32 * wobble).round() as u64),
    )
}
