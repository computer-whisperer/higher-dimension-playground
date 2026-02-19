use super::*;

pub(super) fn sanitize_player_name(name: &str, client_id: u64) -> String {
    let trimmed = name.trim();
    if trimmed.is_empty() {
        return format!("player-{client_id}");
    }
    trimmed.chars().take(32).collect()
}

fn normalize_spawn_token(token: &str) -> String {
    token
        .chars()
        .filter(|ch| ch.is_ascii_alphanumeric())
        .map(|ch| ch.to_ascii_lowercase())
        .collect()
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

pub(super) fn parse_spawn_material_id(token: &str) -> Option<u8> {
    if let Ok(id) = token.parse::<u8>() {
        if (1..=materials::MAX_MATERIAL_ID).contains(&id) {
            return Some(id);
        }
    }
    let normalized = normalize_spawn_token(token);
    materials::MATERIALS.iter().find_map(|material| {
        (normalize_spawn_token(material.name) == normalized).then_some(material.id)
    })
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
    let names = SPAWNABLE_ENTITY_SPECS
        .iter()
        .map(|spec| spec.canonical_name)
        .collect::<Vec<_>>()
        .join("|");
    format!("Usage: /spawn <{names}> [x y z w] [material-id|material-name]")
}

pub(super) fn spawnable_entity_spec_for_token(token: &str) -> Option<SpawnableEntitySpec> {
    let normalized = normalize_spawn_token(token);
    SPAWNABLE_ENTITY_SPECS.iter().copied().find(|spec| {
        spec.aliases
            .iter()
            .any(|alias| normalize_spawn_token(alias) == normalized)
    })
}

pub(super) fn spawnable_entity_spec_for_kind(kind: EntityKind) -> Option<SpawnableEntitySpec> {
    SPAWNABLE_ENTITY_SPECS
        .iter()
        .copied()
        .find(|spec| spec.kind == kind)
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
    let look = normalize4_or_default(snapshot.orientation, [0.0, 0.0, 1.0, 0.0]);
    let position = [
        snapshot.position[0] + look[0] * 3.0,
        snapshot.position[1] + look[1] * 3.0,
        snapshot.position[2] + look[2] * 3.0,
        snapshot.position[3] + look[3] * 3.0,
    ];
    (position, look)
}

pub(super) fn mob_archetype_defaults(archetype: MobArchetype) -> (f32, f32, f32) {
    match archetype {
        MobArchetype::Seeker => (2.9, 2.6, 0.64),
        MobArchetype::Creeper4d => (2.4, 3.8, 1.15),
        MobArchetype::PhaseSpider => (3.1, 2.4, 0.95),
    }
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
