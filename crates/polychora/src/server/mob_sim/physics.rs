use super::super::*;
use crate::shared::region_tree::ChunkKey;
use crate::shared::spatial::ChunkCoord;

pub(super) fn mob_collision_radius_for_scale(scale: f32) -> f32 {
    let clamped_scale = if scale.is_finite() { scale } else { 0.5 };
    (clamped_scale * MOB_COLLISION_RADIUS_SCALE)
        .clamp(MOB_COLLISION_RADIUS_MIN, MOB_COLLISION_RADIUS_MAX)
}

fn resolved_payload_is_all_air(
    payload: &crate::shared::chunk_payload::ResolvedChunkPayload,
) -> bool {
    use crate::shared::chunk_payload::ChunkPayload;
    match &payload.payload {
        ChunkPayload::Empty | ChunkPayload::Virgin => true,
        ChunkPayload::Uniform(idx) => payload
            .block_palette
            .get(*idx as usize)
            .map(|b| b.is_air())
            .unwrap_or(true),
        _ => false,
    }
}

pub(super) fn sample_effective_voxel_for_collision(
    state: &ServerState,
    cache: &mut HashMap<ChunkKey, CollisionChunkCacheEntry>,
    wx: i32,
    wy: i32,
    wz: i32,
    ww: i32,
) -> bool {
    let (chunk_pos, voxel_index) = voxel::world_to_chunk_at_scale(
        ChunkCoord::from_num(wx),
        ChunkCoord::from_num(wy),
        ChunkCoord::from_num(wz),
        ChunkCoord::from_num(ww),
        0,
    );
    if let Some(entry) = cache.get(&chunk_pos) {
        return match entry {
            CollisionChunkCacheEntry::Explicit(payload) => !payload.block_at(voxel_index).is_air(),
            CollisionChunkCacheEntry::ExplicitEmpty => false,
            CollisionChunkCacheEntry::Effective(payload) => payload
                .as_ref()
                .map(|p| !p.block_at(voxel_index).is_air())
                .unwrap_or(false),
        };
    }

    if let Some((payload, _scale)) = state.world_chunk_at(chunk_pos) {
        if resolved_payload_is_all_air(&payload) {
            cache.insert(chunk_pos, CollisionChunkCacheEntry::ExplicitEmpty);
            return false;
        }
        let is_solid = !payload.block_at(voxel_index).is_air();
        cache.insert(chunk_pos, CollisionChunkCacheEntry::Explicit(payload));
        return is_solid;
    }

    if let Some((payload, _scale)) = state.cached_chunk_at(chunk_pos) {
        if resolved_payload_is_all_air(&payload) {
            cache.insert(chunk_pos, CollisionChunkCacheEntry::ExplicitEmpty);
            return false;
        }
        let is_solid = !payload.block_at(voxel_index).is_air();
        cache.insert(chunk_pos, CollisionChunkCacheEntry::Explicit(payload));
        return is_solid;
    }

    let effective_chunk = state.world_effective_chunk(chunk_pos, false);
    let is_solid = effective_chunk
        .as_ref()
        .map(|p| !p.block_at(voxel_index).is_air())
        .unwrap_or(false);
    cache.insert(
        chunk_pos,
        CollisionChunkCacheEntry::Effective(effective_chunk),
    );
    is_solid
}

pub(super) fn mob_collides_at(
    state: &ServerState,
    cache: &mut HashMap<ChunkKey, CollisionChunkCacheEntry>,
    position: [f32; 4],
    radius: f32,
) -> bool {
    mob_collides_at_with_bounds(state, cache, position, radius, &state.world_bounds())
}

fn mob_collides_at_with_bounds(
    state: &ServerState,
    cache: &mut HashMap<ChunkKey, CollisionChunkCacheEntry>,
    position: [f32; 4],
    radius: f32,
    world_bounds: &WorldBounds,
) -> bool {
    for (axis, &pos) in position.iter().enumerate() {
        if let Some(lo) = world_bounds.min[axis] {
            if pos - radius < lo {
                return true;
            }
        }
        if let Some(hi) = world_bounds.max[axis] {
            if pos + radius > hi {
                return true;
            }
        }
    }

    let min = [
        position[0] - radius,
        position[1] - radius,
        position[2] - radius,
        position[3] - radius,
    ];
    let max = [
        position[0] + radius,
        position[1] + radius,
        position[2] + radius,
        position[3] + radius,
    ];
    let max_epsilon = 1e-4f32;
    let lo = [
        min[0].floor() as i32,
        min[1].floor() as i32,
        min[2].floor() as i32,
        min[3].floor() as i32,
    ];
    let hi = [
        (max[0] - max_epsilon).floor() as i32,
        (max[1] - max_epsilon).floor() as i32,
        (max[2] - max_epsilon).floor() as i32,
        (max[3] - max_epsilon).floor() as i32,
    ];
    if hi[0] < lo[0] || hi[1] < lo[1] || hi[2] < lo[2] || hi[3] < lo[3] {
        return false;
    }

    for x in lo[0]..=hi[0] {
        for y in lo[1]..=hi[1] {
            for z in lo[2]..=hi[2] {
                for w in lo[3]..=hi[3] {
                    if sample_effective_voxel_for_collision(state, cache, x, y, z, w) {
                        return true;
                    }
                }
            }
        }
    }
    false
}

pub(super) fn resolve_mob_collision(
    state: &ServerState,
    cache: &mut HashMap<ChunkKey, CollisionChunkCacheEntry>,
    old_pos: [f32; 4],
    attempted_pos: [f32; 4],
    scale: f32,
) -> [f32; 4] {
    let radius = mob_collision_radius_for_scale(scale);
    let mut pos = old_pos;
    if mob_collides_at(state, cache, pos, radius) {
        for _ in 0..MOB_COLLISION_MAX_PUSHUP_STEPS {
            pos[1] += MOB_COLLISION_PUSHUP_STEP;
            if !mob_collides_at(state, cache, pos, radius) {
                break;
            }
        }
    }

    for axis in [0usize, 2, 3, 1] {
        let target = attempted_pos[axis];
        if (target - pos[axis]).abs() <= 1e-6 {
            continue;
        }

        let mut candidate = pos;
        candidate[axis] = target;
        if !mob_collides_at(state, cache, candidate, radius) {
            pos = candidate;
            continue;
        }

        let mut feasible = pos[axis];
        let mut blocked = target;
        for _ in 0..MOB_COLLISION_BINARY_STEPS {
            let mid = 0.5 * (feasible + blocked);
            let mut probe = pos;
            probe[axis] = mid;
            if mob_collides_at(state, cache, probe, radius) {
                blocked = mid;
            } else {
                feasible = mid;
            }
        }
        pos[axis] = feasible;
    }

    pos
}

fn mob_direction_for_locomotion(direction: [f32; 4], locomotion: MobLocomotionMode) -> [f32; 4] {
    match locomotion {
        MobLocomotionMode::Walking => normalize4_or_default(
            [direction[0], 0.0, direction[2], direction[3]],
            [0.0, 0.0, 1.0, 0.0],
        ),
        MobLocomotionMode::Flying => normalize4_or_default(direction, [0.0, 0.0, 1.0, 0.0]),
    }
}

pub(super) fn integrate_mob_steering_step(
    mob: &MobState,
    position: [f32; 4],
    steering: super::MobSteeringCommand,
    locomotion: MobLocomotionMode,
    now_ms: u64,
    previous_update_ms: u64,
) -> ([f32; 4], [f32; 4]) {
    let dt_s = (now_ms.saturating_sub(previous_update_ms)).max(1) as f32 * 0.001;
    let move_dir = mob_direction_for_locomotion(steering.desired_direction, locomotion);
    let step = mob.move_speed.max(0.1) * steering.speed_factor.max(0.0) * dt_s;
    let next_position = [
        position[0] + move_dir[0] * step,
        position[1] + move_dir[1] * step,
        position[2] + move_dir[2] * step,
        position[3] + move_dir[3] * step,
    ];
    (next_position, move_dir)
}

fn mob_horizontal_distance_sq(a: [f32; 4], b: [f32; 4]) -> f32 {
    let dx = a[0] - b[0];
    let dz = a[2] - b[2];
    let dw = a[3] - b[3];
    dx * dx + dz * dz + dw * dw
}

pub(super) fn stick_mob_to_ground(
    state: &ServerState,
    cache: &mut HashMap<ChunkKey, CollisionChunkCacheEntry>,
    position: [f32; 4],
    scale: f32,
    max_drop: f32,
) -> [f32; 4] {
    if !max_drop.is_finite() || max_drop <= 0.0 {
        return position;
    }
    let radius = mob_collision_radius_for_scale(scale);
    let mut pos = position;
    let mut dropped = 0.0f32;
    while dropped + MOB_WALK_GROUND_STICK_STEP <= max_drop {
        let mut probe = pos;
        probe[1] -= MOB_WALK_GROUND_STICK_STEP;
        if mob_collides_at(state, cache, probe, radius) {
            break;
        }
        pos = probe;
        dropped += MOB_WALK_GROUND_STICK_STEP;
    }
    pos
}

pub(super) fn resolve_walking_collision_with_steps(
    state: &ServerState,
    cache: &mut HashMap<ChunkKey, CollisionChunkCacheEntry>,
    old_position: [f32; 4],
    attempted_position: [f32; 4],
    scale: f32,
) -> [f32; 4] {
    let resolved = resolve_mob_collision(state, cache, old_position, attempted_position, scale);
    let mut best = stick_mob_to_ground(
        state,
        cache,
        resolved,
        scale,
        MOB_WALK_GROUND_STICK_MAX_DROP,
    );
    let desired_progress_sq = mob_horizontal_distance_sq(old_position, attempted_position);
    if desired_progress_sq <= 1e-6 {
        return best;
    }

    let mut best_progress_sq = mob_horizontal_distance_sq(old_position, best);
    let mut lift = MOB_WALK_STEP_UP_SAMPLE_STEP;
    while lift <= MOB_WALK_STEP_UP_MAX_HEIGHT + 1e-5 {
        let raised_old = [
            old_position[0],
            old_position[1] + lift,
            old_position[2],
            old_position[3],
        ];
        let raised_attempted = [
            attempted_position[0],
            attempted_position[1] + lift,
            attempted_position[2],
            attempted_position[3],
        ];
        let raised_resolved =
            resolve_mob_collision(state, cache, raised_old, raised_attempted, scale);
        let grounded = stick_mob_to_ground(
            state,
            cache,
            raised_resolved,
            scale,
            MOB_WALK_GROUND_STICK_MAX_DROP + lift,
        );
        let progress_sq = mob_horizontal_distance_sq(old_position, grounded);
        if progress_sq > best_progress_sq + 1e-5 {
            best = grounded;
            best_progress_sq = progress_sq;
            if best_progress_sq + 1e-5 >= desired_progress_sq {
                break;
            }
        }
        lift += MOB_WALK_STEP_UP_SAMPLE_STEP;
    }

    best
}

pub(super) fn apply_mob_locomotion_post_collision(
    state: &ServerState,
    cache: &mut HashMap<ChunkKey, CollisionChunkCacheEntry>,
    old_position: [f32; 4],
    attempted_position: [f32; 4],
    scale: f32,
    locomotion: MobLocomotionMode,
) -> [f32; 4] {
    match locomotion {
        MobLocomotionMode::Walking => resolve_walking_collision_with_steps(
            state,
            cache,
            old_position,
            attempted_position,
            scale,
        ),
        MobLocomotionMode::Flying => {
            resolve_mob_collision(state, cache, old_position, attempted_position, scale)
        }
    }
}
