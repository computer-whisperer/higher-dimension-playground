use super::runtime_net::{apply_creeper_explosion, apply_explosion_impulse};
use super::*;
use crate::shared::region_tree::ChunkKey;
use crate::shared::wasm::{WasmPluginManager, WasmPluginSlot};
use polychora_plugin_api::entity_tick_abi::{
    EntityAbilityCheck, EntityAbilityResult, EntityTickInput, EntityTickOutput,
};
use polychora_plugin_api::opcodes::{OP_ENTITY_ABILITY, OP_ENTITY_TICK};

#[derive(Clone, Debug, Default)]
pub(super) struct SimTimings {
    pub(super) sim_steps: usize,
    pub(super) wasm_us: u64,
    pub(super) nav_us: u64,
    pub(super) collision_us: u64,
}

fn sanitize_direction(d: [f32; 4]) -> Option<[f32; 4]> {
    if d.iter().all(|v| v.is_finite()) {
        Some(d)
    } else {
        None
    }
}

fn wasm_entity_steer(
    manager: &mut WasmPluginManager,
    input: &EntityTickInput,
) -> Option<MobSteeringCommand> {
    let input_bytes = postcard::to_allocvec(input).ok()?;
    let result = manager
        .call_slot(WasmPluginSlot::EntitySimulation, OP_ENTITY_TICK as i32, &input_bytes)
        .ok()??;
    let output: EntityTickOutput = postcard::from_bytes(&result.invocation.output).ok()?;
    match output {
        EntityTickOutput::Steer { desired_direction, speed_factor } => {
            let desired_direction = sanitize_direction(desired_direction)?;
            let speed_factor = if speed_factor.is_finite() {
                speed_factor.clamp(0.0, 10.0)
            } else {
                return None;
            };
            Some(MobSteeringCommand {
                desired_direction,
                speed_factor,
            })
        }
        EntityTickOutput::SetPose { .. } => None, // PhysicsDriven mobs should not return SetPose
    }
}

fn wasm_entity_ability(
    manager: &mut WasmPluginManager,
    check: &EntityAbilityCheck,
) -> Option<EntityAbilityResult> {
    let input_bytes = postcard::to_allocvec(check).ok()?;
    let result = manager
        .call_slot(
            WasmPluginSlot::EntitySimulation,
            OP_ENTITY_ABILITY as i32,
            &input_bytes,
        )
        .ok()??;
    postcard::from_bytes(&result.invocation.output).ok()
}

fn mob_collision_radius_for_scale(scale: f32) -> f32 {
    let clamped_scale = if scale.is_finite() { scale } else { 0.5 };
    (clamped_scale * MOB_COLLISION_RADIUS_SCALE)
        .clamp(MOB_COLLISION_RADIUS_MIN, MOB_COLLISION_RADIUS_MAX)
}

#[derive(Clone, Copy, Debug)]
struct MobSteeringCommand {
    desired_direction: [f32; 4],
    speed_factor: f32,
}

fn resolved_payload_is_all_air(payload: &crate::shared::chunk_payload::ResolvedChunkPayload) -> bool {
    use crate::shared::chunk_payload::ChunkPayload;
    match &payload.payload {
        ChunkPayload::Empty => true,
        ChunkPayload::Uniform(idx) => payload
            .block_palette
            .get(*idx as usize)
            .map(|b| b.is_air())
            .unwrap_or(true),
        _ => false,
    }
}

fn sample_effective_voxel_for_collision(
    state: &ServerState,
    cache: &mut HashMap<ChunkKey, CollisionChunkCacheEntry>,
    wx: i32,
    wy: i32,
    wz: i32,
    ww: i32,
) -> bool {
    let (chunk_pos, voxel_index) = voxel::world_to_chunk(wx, wy, wz, ww);
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

    if let Some(payload) = state.world_chunk_at(chunk_pos) {
        if resolved_payload_is_all_air(&payload) {
            cache.insert(chunk_pos, CollisionChunkCacheEntry::ExplicitEmpty);
            return false;
        }
        let is_solid = !payload.block_at(voxel_index).is_air();
        cache.insert(chunk_pos, CollisionChunkCacheEntry::Explicit(payload));
        return is_solid;
    }

    if let Some(payload) = state.mob_nav_cached_chunk_at(chunk_pos) {
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

fn mob_collides_at(
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
    for axis in 0..4 {
        if let Some(lo) = world_bounds.min[axis] {
            if position[axis] - radius < lo {
                return true;
            }
        }
        if let Some(hi) = world_bounds.max[axis] {
            if position[axis] + radius > hi {
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

fn resolve_mob_collision(
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

fn nearest_position_to(position: [f32; 4], candidates: &[[f32; 4]]) -> Option<[f32; 4]> {
    candidates.iter().copied().min_by(|a, b| {
        distance4_sq(*a, position)
            .partial_cmp(&distance4_sq(*b, position))
            .unwrap_or(std::cmp::Ordering::Equal)
    })
}

fn mob_nav_base_cell_from_position(
    position: [f32; 4],
    locomotion: MobLocomotionMode,
) -> MobNavCell {
    let y = match locomotion {
        MobLocomotionMode::Walking | MobLocomotionMode::Flying => position[1].ceil() as i32,
    };
    [
        position[0].round() as i32,
        y,
        position[2].round() as i32,
        position[3].round() as i32,
    ]
}

fn mob_nav_position_from_cell(cell: MobNavCell) -> [f32; 4] {
    [
        cell[0] as f32,
        cell[1] as f32,
        cell[2] as f32,
        cell[3] as f32,
    ]
}

fn mob_nav_manhattan_distance(a: MobNavCell, b: MobNavCell) -> i32 {
    let dx = a[0].abs_diff(b[0]) as i32;
    let dy = a[1].abs_diff(b[1]) as i32;
    let dz = a[2].abs_diff(b[2]) as i32;
    let dw = a[3].abs_diff(b[3]) as i32;
    dx.saturating_add(dy).saturating_add(dz).saturating_add(dw)
}

fn mob_nav_has_line_of_sight(
    state: &ServerState,
    cache: &mut HashMap<ChunkKey, CollisionChunkCacheEntry>,
    from: [f32; 4],
    to: [f32; 4],
    collision_radius: f32,
    locomotion: MobLocomotionMode,
) -> bool {
    let delta = [
        to[0] - from[0],
        to[1] - from[1],
        to[2] - from[2],
        to[3] - from[3],
    ];
    let dist_sq = distance4_sq(from, to);
    if dist_sq <= 1e-6 {
        return true;
    }
    let dist = dist_sq.sqrt();
    let steps = (dist / MOB_NAV_PATH_LOS_STEP).ceil().max(1.0).min(256.0) as usize;
    for idx in 1..=steps {
        let t = idx as f32 / steps as f32;
        let probe = [
            from[0] + delta[0] * t,
            from[1] + delta[1] * t,
            from[2] + delta[2] * t,
            from[3] + delta[3] * t,
        ];
        if mob_collides_at(state, cache, probe, collision_radius) {
            return false;
        }
        if locomotion == MobLocomotionMode::Walking {
            let probe_cell = mob_nav_base_cell_from_position(probe, locomotion);
            if !mob_nav_cell_is_walkable(
                state,
                cache,
                probe_cell,
                collision_radius,
                locomotion,
            ) {
                return false;
            }
        }
    }
    true
}

fn mob_nav_cell_is_walkable(
    state: &ServerState,
    cache: &mut HashMap<ChunkKey, CollisionChunkCacheEntry>,
    cell: MobNavCell,
    collision_radius: f32,
    locomotion: MobLocomotionMode,
) -> bool {
    if mob_collides_at(
        state,
        cache,
        mob_nav_position_from_cell(cell),
        collision_radius,
    ) {
        return false;
    }
    if locomotion == MobLocomotionMode::Walking {
        let has_support = sample_effective_voxel_for_collision(
            state,
            cache,
            cell[0],
            cell[1] - 2,
            cell[2],
            cell[3],
        );
        if !has_support {
            return false;
        }
    }
    true
}

fn mob_nav_neighbor_steps(locomotion: MobLocomotionMode) -> &'static [MobNavCell] {
    const FLYING_STEPS: [MobNavCell; 8] = [
        [1, 0, 0, 0],
        [-1, 0, 0, 0],
        [0, 1, 0, 0],
        [0, -1, 0, 0],
        [0, 0, 1, 0],
        [0, 0, -1, 0],
        [0, 0, 0, 1],
        [0, 0, 0, -1],
    ];
    const WALKING_STEPS: [MobNavCell; 18] = [
        [1, 0, 0, 0],
        [1, 1, 0, 0],
        [1, -1, 0, 0],
        [-1, 0, 0, 0],
        [-1, 1, 0, 0],
        [-1, -1, 0, 0],
        [0, 0, 1, 0],
        [0, 1, 1, 0],
        [0, -1, 1, 0],
        [0, 0, -1, 0],
        [0, 1, -1, 0],
        [0, -1, -1, 0],
        [0, 0, 0, 1],
        [0, 1, 0, 1],
        [0, -1, 0, 1],
        [0, 0, 0, -1],
        [0, 1, 0, -1],
        [0, -1, 0, -1],
    ];
    match locomotion {
        MobLocomotionMode::Walking => &WALKING_STEPS,
        MobLocomotionMode::Flying => &FLYING_STEPS,
    }
}

fn mob_nav_find_walkable_goal_cell(
    state: &ServerState,
    cache: &mut HashMap<ChunkKey, CollisionChunkCacheEntry>,
    desired_goal: MobNavCell,
    origin: MobNavCell,
    collision_radius: f32,
    locomotion: MobLocomotionMode,
) -> Option<MobNavCell> {
    if mob_nav_cell_is_walkable(
        state,
        cache,
        desired_goal,
        collision_radius,
        locomotion,
    ) {
        return Some(desired_goal);
    }

    let max_radius = MOB_NAV_PATH_GOAL_ADJUST_RADIUS_CELLS.max(0);
    let mut best: Option<(i32, MobNavCell)> = None;
    for dx in -max_radius..=max_radius {
        for dy in -max_radius..=max_radius {
            for dz in -max_radius..=max_radius {
                for dw in -max_radius..=max_radius {
                    let ring = dx.abs() + dy.abs() + dz.abs() + dw.abs();
                    if ring == 0 || ring > max_radius {
                        continue;
                    }
                    let candidate = [
                        desired_goal[0] + dx,
                        desired_goal[1] + dy,
                        desired_goal[2] + dz,
                        desired_goal[3] + dw,
                    ];
                    if !mob_nav_cell_is_walkable(
                        state,
                        cache,
                        candidate,
                        collision_radius,
                        locomotion,
                    ) {
                        continue;
                    }
                    let score = ring
                        .saturating_mul(16)
                        .saturating_add(mob_nav_manhattan_distance(candidate, origin));
                    match best {
                        Some((best_score, _)) if score >= best_score => {}
                        _ => best = Some((score, candidate)),
                    }
                }
            }
        }
    }
    best.map(|(_, cell)| cell)
}

fn mob_nav_reconstruct_path(
    start: MobNavCell,
    goal: MobNavCell,
    came_from: &HashMap<MobNavCell, MobNavCell>,
) -> Option<Vec<MobNavCell>> {
    if start == goal {
        return Some(Vec::new());
    }
    let mut reverse = Vec::new();
    let mut cursor = goal;
    let mut guards = 0usize;
    while cursor != start {
        reverse.push(cursor);
        cursor = *came_from.get(&cursor)?;
        guards = guards.saturating_add(1);
        if guards > MOB_NAV_PATH_MAX_SEARCH_STEPS {
            return None;
        }
    }
    reverse.reverse();
    if reverse.len() > MOB_NAV_PATH_MAX_WAYPOINTS {
        reverse.truncate(MOB_NAV_PATH_MAX_WAYPOINTS);
    }
    Some(reverse)
}

fn mob_nav_find_path(
    state: &ServerState,
    cache: &mut HashMap<ChunkKey, CollisionChunkCacheEntry>,
    start: MobNavCell,
    goal: MobNavCell,
    collision_radius: f32,
    locomotion: MobLocomotionMode,
) -> Option<MobNavPathResult> {
    if start == goal {
        return Some(MobNavPathResult {
            path_cells: Vec::new(),
            reached_goal: true,
            expanded_steps: 0,
            best_cell: start,
            best_goal_distance: 0,
        });
    }
    if !mob_nav_cell_is_walkable(state, cache, start, collision_radius, locomotion) {
        return None;
    }
    if !mob_nav_cell_is_walkable(state, cache, goal, collision_radius, locomotion) {
        return None;
    }

    let mut open = BinaryHeap::<(Reverse<i32>, Reverse<i32>, MobNavCell)>::new();
    let mut g_scores = HashMap::<MobNavCell, i32>::new();
    let mut came_from = HashMap::<MobNavCell, MobNavCell>::new();
    let mut best_cell = start;
    let mut best_h = mob_nav_manhattan_distance(start, goal);
    let mut best_g = 0i32;

    g_scores.insert(start, 0);
    open.push((Reverse(best_h), Reverse(0), start));

    let mut visited_steps = 0usize;
    while let Some((_f_score, Reverse(g_cost), cell)) = open.pop() {
        let current_best = g_scores.get(&cell).copied().unwrap_or(i32::MAX);
        if g_cost > current_best {
            continue;
        }
        let h_cost = mob_nav_manhattan_distance(cell, goal);
        if h_cost < best_h || (h_cost == best_h && g_cost < best_g) {
            best_cell = cell;
            best_h = h_cost;
            best_g = g_cost;
        }
        if cell == goal {
            let path_cells = mob_nav_reconstruct_path(start, goal, &came_from)?;
            return Some(MobNavPathResult {
                path_cells,
                reached_goal: true,
                expanded_steps: visited_steps,
                best_cell: goal,
                best_goal_distance: 0,
            });
        }

        visited_steps = visited_steps.saturating_add(1);
        if visited_steps > MOB_NAV_PATH_MAX_SEARCH_STEPS {
            break;
        }

        for step in mob_nav_neighbor_steps(locomotion) {
            let next = [
                cell[0] + step[0],
                cell[1] + step[1],
                cell[2] + step[2],
                cell[3] + step[3],
            ];
            if mob_nav_manhattan_distance(start, next) > MOB_NAV_PATH_MAX_SEARCH_RADIUS_CELLS {
                continue;
            }

            let tentative_g = g_cost.saturating_add(1);
            let known_next_g = g_scores.get(&next).copied().unwrap_or(i32::MAX);
            if tentative_g >= known_next_g {
                continue;
            }
            if !mob_nav_cell_is_walkable(state, cache, next, collision_radius, locomotion) {
                continue;
            }

            came_from.insert(next, cell);
            g_scores.insert(next, tentative_g);
            let h_cost = mob_nav_manhattan_distance(next, goal);
            open.push((
                Reverse(tentative_g.saturating_add(h_cost)),
                Reverse(tentative_g),
                next,
            ));
        }
    }

    if best_cell != start {
        let path_cells = mob_nav_reconstruct_path(start, best_cell, &came_from)?;
        return Some(MobNavPathResult {
            path_cells,
            reached_goal: false,
            expanded_steps: visited_steps,
            best_cell,
            best_goal_distance: best_h,
        });
    }
    None
}

fn mob_nav_debug_log(
    navigation: &mut MobNavigationState,
    debug_enabled: bool,
    now_ms: u64,
    mob_entity_id: u64,
    entity_ns: u32,
    entity_type: u32,
    message: &str,
) {
    if !debug_enabled {
        return;
    }
    if now_ms.saturating_sub(navigation.last_debug_log_ms) < MOB_NAV_DEBUG_MIN_INTERVAL_MS {
        return;
    }
    navigation.last_debug_log_ms = now_ms;
    eprintln!(
        "[mob-nav][server] t={} entity={} type=({:#x},{:#x}) {}",
        now_ms, mob_entity_id, entity_ns, entity_type, message
    );
}

fn update_mob_navigation_state(
    state: &ServerState,
    cache: &mut HashMap<ChunkKey, CollisionChunkCacheEntry>,
    mut navigation: MobNavigationState,
    mob_entity_id: u64,
    entity_ns: u32,
    entity_type: u32,
    locomotion: MobLocomotionMode,
    debug_enabled: bool,
    position: [f32; 4],
    scale: f32,
    target_position: Option<[f32; 4]>,
    now_ms: u64,
) -> (Option<[f32; 4]>, bool, MobNavigationState) {
    let Some(target) = target_position else {
        navigation.goal_cell = None;
        navigation.path_cells.clear();
        navigation.path_cursor = 0;
        navigation.blocked_without_path = false;
        return (None, false, navigation);
    };

    let collision_radius = mob_collision_radius_for_scale(scale);
    let direct_target = match locomotion {
        MobLocomotionMode::Walking => [target[0], position[1], target[2], target[3]],
        MobLocomotionMode::Flying => target,
    };
    let desired_start_cell = mob_nav_base_cell_from_position(position, locomotion);
    let start_cell = mob_nav_find_walkable_goal_cell(
        state,
        cache,
        desired_start_cell,
        desired_start_cell,
        collision_radius,
        locomotion,
    )
    .unwrap_or(desired_start_cell);
    let desired_goal_cell = mob_nav_base_cell_from_position(target, locomotion);
    let goal_changed = navigation
        .goal_cell
        .map(|goal| {
            mob_nav_manhattan_distance(goal, desired_goal_cell)
                > MOB_NAV_PATH_GOAL_REPLAN_THRESHOLD_CELLS
        })
        .unwrap_or(true);
    let path_exhausted = navigation.path_cursor >= navigation.path_cells.len();
    let repath_due =
        now_ms.saturating_sub(navigation.last_repath_ms) >= MOB_NAV_PATH_REPLAN_INTERVAL_MS;

    let has_los = if !goal_changed && now_ms.saturating_sub(navigation.last_los_check_ms) < MOB_NAV_LOS_CACHE_MS {
        navigation.last_los_result
    } else {
        let result = mob_nav_has_line_of_sight(state, cache, position, direct_target, collision_radius, locomotion);
        navigation.last_los_result = result;
        navigation.last_los_check_ms = now_ms;
        result
    };
    if has_los {
        navigation.goal_cell = Some(desired_goal_cell);
        navigation.path_cells.clear();
        navigation.path_cursor = 0;
        navigation.blocked_without_path = false;
        if goal_changed || path_exhausted || repath_due {
            mob_nav_debug_log(
                &mut navigation,
                debug_enabled,
                now_ms,
                mob_entity_id,
                entity_ns,
                entity_type,
                &format!(
                    "mode=los start={:?} goal={:?} path_exhausted={} repath_due={}",
                    start_cell, desired_goal_cell, path_exhausted, repath_due
                ),
            );
        }
        return (Some(direct_target), false, navigation);
    }

    if goal_changed || path_exhausted || repath_due {
        let goal_cell = mob_nav_find_walkable_goal_cell(
            state,
            cache,
            desired_goal_cell,
            start_cell,
            collision_radius,
            locomotion,
        )
        .unwrap_or(desired_goal_cell);
        navigation.goal_cell = Some(goal_cell);
        navigation.last_repath_ms = now_ms;

        if let Some(path_result) =
            mob_nav_find_path(state, cache, start_cell, goal_cell, collision_radius, locomotion)
        {
            let path_len = path_result.path_cells.len();
            let reached_goal = path_result.reached_goal;
            let expanded_steps = path_result.expanded_steps;
            let best_goal_distance = path_result.best_goal_distance;
            let best_cell = path_result.best_cell;
            navigation.path_cells = path_result.path_cells;
            navigation.path_cursor = 0;
            navigation.blocked_without_path = false;
            mob_nav_debug_log(
                &mut navigation,
                debug_enabled,
                now_ms,
                mob_entity_id,
                entity_ns,
                entity_type,
                &format!(
                    "mode=path start={:?} goal={:?} reached_goal={} path_len={} expanded={} best_cell={:?} best_goal_dist={}",
                    start_cell,
                    goal_cell,
                    reached_goal,
                    path_len,
                    expanded_steps,
                    best_cell,
                    best_goal_distance
                ),
            );
        } else if goal_changed || path_exhausted {
            navigation.path_cells.clear();
            navigation.path_cursor = 0;
            navigation.blocked_without_path = true;
            mob_nav_debug_log(
                &mut navigation,
                debug_enabled,
                now_ms,
                mob_entity_id,
                entity_ns,
                entity_type,
                &format!(
                    "mode=path-fail start={:?} goal={:?} (no reachable cell found, fallback=wander)",
                    start_cell,
                    goal_cell
                ),
            );
        }
    }

    while navigation.path_cursor < navigation.path_cells.len() {
        let waypoint = mob_nav_position_from_cell(navigation.path_cells[navigation.path_cursor]);
        if distance4_sq(position, waypoint).sqrt() <= MOB_NAV_PATH_NODE_REACH_DISTANCE {
            navigation.path_cursor = navigation.path_cursor.saturating_add(1);
        } else {
            break;
        }
    }

    if navigation.path_cursor < navigation.path_cells.len() {
        let waypoint = mob_nav_position_from_cell(navigation.path_cells[navigation.path_cursor]);
        (Some(waypoint), true, navigation)
    } else if navigation.blocked_without_path {
        (None, false, navigation)
    } else {
        (Some(direct_target), false, navigation)
    }
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

fn integrate_mob_steering_step(
    mob: &MobState,
    position: [f32; 4],
    steering: MobSteeringCommand,
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

fn stick_mob_to_ground(
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

fn resolve_walking_collision_with_steps(
    state: &ServerState,
    cache: &mut HashMap<ChunkKey, CollisionChunkCacheEntry>,
    old_position: [f32; 4],
    attempted_position: [f32; 4],
    scale: f32,
) -> [f32; 4] {
    let resolved = resolve_mob_collision(state, cache, old_position, attempted_position, scale);
    let mut best = stick_mob_to_ground(state, cache, resolved, scale, MOB_WALK_GROUND_STICK_MAX_DROP);
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
        let raised_resolved = resolve_mob_collision(state, cache, raised_old, raised_attempted, scale);
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

fn apply_mob_locomotion_post_collision(
    state: &ServerState,
    cache: &mut HashMap<ChunkKey, CollisionChunkCacheEntry>,
    old_position: [f32; 4],
    attempted_position: [f32; 4],
    scale: f32,
    locomotion: MobLocomotionMode,
) -> [f32; 4] {
    match locomotion {
        MobLocomotionMode::Walking => {
            resolve_walking_collision_with_steps(state, cache, old_position, attempted_position, scale)
        }
        MobLocomotionMode::Flying => {
            resolve_mob_collision(state, cache, old_position, attempted_position, scale)
        }
    }
}

fn attempt_phase_spider_blink(
    state: &ServerState,
    cache: &mut HashMap<ChunkKey, CollisionChunkCacheEntry>,
    position: [f32; 4],
    forward: [f32; 4],
    scale: f32,
    phase_offset: f32,
    now_ms: u64,
    blink_distance: f32,
    blink_min_distance: f32,
) -> Option<[f32; 4]> {
    let forward = normalize4_or_default(forward, [0.0, 0.0, 1.0, 0.0]);
    let phase = now_ms as f32 * 0.0047 + phase_offset * 1.3;
    let strafe_a = normalize4_or_default(
        [-forward[2], forward[3], forward[0], -forward[1]],
        [forward[3], 0.0, -forward[0], forward[1]],
    );
    let strafe_b = normalize4_or_default(
        [forward[1], -forward[0], forward[3], -forward[2]],
        [0.0, forward[2], -forward[1], forward[0]],
    );
    let drift = normalize4_or_default(
        [
            strafe_a[0] * phase.sin() + strafe_b[0] * phase.cos(),
            strafe_a[1] * phase.sin() + strafe_b[1] * phase.cos(),
            strafe_a[2] * phase.sin() + strafe_b[2] * phase.cos(),
            strafe_a[3] * phase.sin() + strafe_b[3] * phase.cos(),
        ],
        strafe_a,
    );
    let blink_dir = normalize4_or_default(
        [
            forward[0] * 1.0 + drift[0] * 0.42,
            forward[1] * 1.0 + drift[1] * 0.42,
            forward[2] * 1.0 + drift[2] * 0.42,
            forward[3] * 1.0 + drift[3] * 0.42,
        ],
        forward,
    );

    let radius = mob_collision_radius_for_scale(scale);
    let lift_primary = 0.16 + 0.12 * (phase * 0.8).sin().abs();
    let lift_options = [lift_primary, 0.05, -0.08];

    let mut distance = blink_distance;
    while distance >= blink_min_distance {
        for &lift in &lift_options {
            let candidate = [
                position[0] + blink_dir[0] * distance,
                position[1] + blink_dir[1] * distance + lift,
                position[2] + blink_dir[2] * distance,
                position[3] + blink_dir[3] * distance,
            ];
            if !mob_collides_at(state, cache, candidate, radius) {
                return Some(candidate);
            }
        }
        distance -= 0.55;
    }
    None
}

fn simulate_mobs(
    state: &mut ServerState,
    collision_cache: &mut HashMap<ChunkKey, CollisionChunkCacheEntry>,
    wasm_manager: &mut Option<WasmPluginManager>,
    now_ms: u64,
    timings: &mut SimTimings,
) -> (Vec<QueuedExplosionEvent>, Vec<QueuedPlayerMovementModifier>) {
    let player_positions: Vec<[f32; 4]> = state
        .players
        .values()
        .filter_map(|player| state.entity_store.snapshot(player.entity_id))
        .map(|snapshot| snapshot.entity.pose.position)
        .collect();

    let mut stale = Vec::new();
    let mut detonations: Vec<(u64, [f32; 4], polychora_plugin_api::entity::EntitySimConfig)> = Vec::new();
    let mut navigation_updates = Vec::with_capacity(state.mobs.len());
    let mut phase_deadline_updates = Vec::with_capacity(state.mobs.len());
    let mut updates = Vec::with_capacity(state.mobs.len());
    let mob_entity_ids: Vec<u64> = state.mobs.keys().copied().collect();
    for entity_id in mob_entity_ids {
        let Some(mob) = state.mobs.get(&entity_id).cloned() else {
            continue;
        };
        let Some(snapshot) = state.entity_store.snapshot(mob.entity_id) else {
            stale.push(mob.entity_id);
            continue;
        };
        let Some(sim_config) = state.content_registry.sim_config(mob.entity_ns, mob.entity_type) else {
            stale.push(mob.entity_id);
            continue;
        };
        let pos = snapshot.entity.pose.position;
        let scl = snapshot.entity.pose.scale;
        let locomotion = sim_config.locomotion;
        let nearest_target = nearest_position_to(pos, &player_positions);

        // Apply nav target Y offset (e.g. creeper pounces below player)
        let navigation_target = if sim_config.nav_target_y_offset != 0.0 {
            nearest_target.map(|target| {
                [
                    target[0],
                    target[1] - sim_config.nav_target_y_offset,
                    target[2],
                    target[3],
                ]
            })
        } else {
            nearest_target
        };

        // Check detonation trigger via WASM
        if let Some(ref ability) = sim_config.ability_params {
            if ability.detonate_trigger_distance > 0.0 {
                let nearest_player_dist = player_positions
                    .iter()
                    .map(|pp| distance4_sq(pos, *pp).sqrt())
                    .fold(f32::MAX, f32::min);
                let detonate_check = EntityAbilityCheck::Detonate {
                    entity_ns: mob.entity_ns,
                    entity_type: mob.entity_type,
                    nearest_player_distance: nearest_player_dist,
                    trigger_distance: ability.detonate_trigger_distance,
                };
                let wasm_t0 = Instant::now();
                let should_detonate = wasm_manager
                    .as_mut()
                    .and_then(|mgr| wasm_entity_ability(mgr, &detonate_check))
                    .map(|r| r.should_trigger)
                    .unwrap_or(false);
                timings.wasm_us += wasm_t0.elapsed().as_micros() as u64;
                if should_detonate {
                    detonations.push((mob.entity_id, pos, sim_config.clone()));
                    continue;
                }
            }
        }

        let nav_t0 = Instant::now();
        let (target_position, path_following, mut navigation) = update_mob_navigation_state(
            state,
            collision_cache,
            mob.navigation.clone(),
            mob.entity_id,
            mob.entity_ns,
            mob.entity_type,
            locomotion,
            state.mob_nav_debug,
            pos,
            scl,
            navigation_target,
            now_ms,
        );
        timings.nav_us += nav_t0.elapsed().as_micros() as u64;

        // Steering dispatch via WASM
        let steering_input = EntityTickInput {
            entity_ns: mob.entity_ns,
            entity_type: mob.entity_type,
            entity_id: mob.entity_id,
            position: pos,
            home_position: snapshot.entity.pose.position,
            scale: scl,
            target_position,
            path_following,
            simple_steer: state.mob_nav_simple_steer,
            now_ms,
            phase_offset: mob.phase_offset,
            preferred_distance: mob.preferred_distance,
            tangent_weight: mob.tangent_weight,
            locomotion,
        };
        let wasm_t0 = Instant::now();
        let Some(steering) = wasm_manager
            .as_mut()
            .and_then(|mgr| wasm_entity_steer(mgr, &steering_input))
        else {
            timings.wasm_us += wasm_t0.elapsed().as_micros() as u64;
            continue;
        };
        timings.wasm_us += wasm_t0.elapsed().as_micros() as u64;
        let (next_position, next_forward) = integrate_mob_steering_step(
            &mob,
            pos,
            steering,
            locomotion,
            now_ms,
            snapshot.last_update_ms,
        );
        let collision_t0 = Instant::now();
        let mut final_position = apply_mob_locomotion_post_collision(
            state,
            collision_cache,
            pos,
            next_position,
            scl,
            locomotion,
        );
        timings.collision_us += collision_t0.elapsed().as_micros() as u64;
        let mut final_forward = next_forward;

        // Blink ability (phase spider) via WASM
        let has_blink = sim_config.ability_params.as_ref()
            .map(|a| a.blink_min_interval_ms > 0)
            .unwrap_or(false);
        if has_blink && now_ms >= mob.next_phase_ms {
            let ability = sim_config.ability_params.as_ref().unwrap();
            let attempted = distance4_sq(pos, next_position).sqrt();
            let resolved = distance4_sq(pos, final_position).sqrt();
            let blink_check = EntityAbilityCheck::Blink {
                entity_ns: mob.entity_ns,
                entity_type: mob.entity_type,
                has_target: nearest_target.is_some(),
                path_following,
                now_ms,
                next_phase_ms: mob.next_phase_ms,
                blocked_progress_epsilon: ability.blink_blocked_progress_epsilon,
                attempted_move_distance: attempted,
                resolved_move_distance: resolved,
            };
            let wasm_t0 = Instant::now();
            let should_phase = wasm_manager
                .as_mut()
                .and_then(|mgr| wasm_entity_ability(mgr, &blink_check))
                .map(|r| r.should_trigger)
                .unwrap_or(false);
            timings.wasm_us += wasm_t0.elapsed().as_micros() as u64;
            if should_phase {
                let collision_t0 = Instant::now();
                let blink_result = attempt_phase_spider_blink(
                    state,
                    collision_cache,
                    pos,
                    next_forward,
                    scl,
                    mob.phase_offset,
                    now_ms,
                    ability.blink_distance,
                    ability.blink_min_distance,
                );
                timings.collision_us += collision_t0.elapsed().as_micros() as u64;
                if let Some(phase_position) = blink_result {
                    final_forward = normalize4_or_default(
                        [
                            phase_position[0] - pos[0],
                            phase_position[1] - pos[1],
                            phase_position[2] - pos[2],
                            phase_position[3] - pos[3],
                        ],
                        next_forward,
                    );
                    final_position = phase_position;
                    mob_nav_debug_log(
                        &mut navigation,
                        state.mob_nav_debug,
                        now_ms,
                        mob.entity_id,
                        mob.entity_ns,
                        mob.entity_type,
                        &format!("mode=phase-blink success pos={:?}", phase_position),
                    );
                } else {
                    mob_nav_debug_log(
                        &mut navigation,
                        state.mob_nav_debug,
                        now_ms,
                        mob.entity_id,
                        mob.entity_ns,
                        mob.entity_type,
                        "mode=phase-blink fail (no valid destination)",
                    );
                }
            }
            let next_deadline = phase_spider_next_phase_deadline(
                now_ms,
                mob.phase_offset,
                (mob.entity_id as u32) ^ (now_ms as u32),
                ability.blink_min_interval_ms,
                ability.blink_max_interval_ms,
            );
            phase_deadline_updates.push((mob.entity_id, next_deadline));
        }
        navigation_updates.push((mob.entity_id, navigation));
        updates.push((mob.entity_id, final_position, final_forward));
    }

    for (entity_id, navigation) in navigation_updates {
        if let Some(mob) = state.mobs.get_mut(&entity_id) {
            mob.navigation = navigation;
        }
    }
    for (entity_id, next_phase_ms) in phase_deadline_updates {
        if let Some(mob) = state.mobs.get_mut(&entity_id) {
            mob.next_phase_ms = next_phase_ms;
        }
    }

    for (entity_id, next_position, next_forward) in updates {
        if !state
            .entity_store
            .set_motion_state(entity_id, next_position, next_forward, now_ms)
        {
            stale.push(entity_id);
            continue;
        }
    }

    stale.sort_unstable();
    stale.dedup();
    for entity_id in stale {
        state.mobs.remove(&entity_id);
        mark_entity_record_despawned(state, entity_id, Some(now_ms));
        let _ = state.entity_store.despawn(entity_id);
    }

    let mut queued_explosions = Vec::new();
    let mut queued_player_modifiers = Vec::new();
    for (entity_id, center, det_config) in detonations {
        if !state.mobs.contains_key(&entity_id) {
            continue;
        }
        let ability = det_config.ability_params.as_ref().unwrap();
        let (changed_chunk_count, explosion) =
            apply_creeper_explosion(state, entity_id, center, ability.detonate_radius_voxels);
        queued_explosions.push(explosion);
        let (_persistent_motion, mut player_modifiers) = apply_explosion_impulse(
            state,
            entity_id,
            center,
            ability.detonate_impulse_radius,
            ability.detonate_max_impulse_distance,
            now_ms,
        );
        eprintln!(
            "[impulse-debug][server] creeper_explosion source_entity={} center={:?} changed_chunks={} player_modifiers={} persistent_motion={}",
            entity_id,
            center,
            changed_chunk_count,
            player_modifiers.len(),
            _persistent_motion
        );
        for modifier in &player_modifiers {
            let delta = modifier.delta_position;
            let delta_len = (delta[0] * delta[0]
                + delta[1] * delta[1]
                + delta[2] * delta[2]
                + delta[3] * delta[3])
                .sqrt();
            eprintln!(
                "[impulse-debug][server] queued_modifier client_id={} source_entity={:?} delta={:?} delta_len={:.3} delta_velocity_y={:.3}",
                modifier.client_id,
                modifier.source_entity_id,
                modifier.delta_position,
                delta_len,
                modifier.delta_velocity_y
            );
        }
        queued_player_modifiers.append(&mut player_modifiers);
        state.mobs.remove(&entity_id);
        mark_entity_record_despawned(state, entity_id, Some(now_ms));
        let _ = state.entity_store.despawn(entity_id);
    }

    (queued_explosions, queued_player_modifiers)
}

pub(super) fn tick_entity_simulation_window(
    state: &mut ServerState,
    wasm_manager: &mut Option<WasmPluginManager>,
    now_ms: u64,
    next_sim_ms: &mut u64,
    sim_step_ms: u64,
) -> (Vec<QueuedExplosionEvent>, Vec<QueuedPlayerMovementModifier>, SimTimings) {
    let mut queued_explosions = Vec::new();
    let mut queued_player_modifiers = Vec::new();
    let mut timings = SimTimings::default();
    let mut collision_cache = HashMap::<ChunkKey, CollisionChunkCacheEntry>::new();
    while *next_sim_ms <= now_ms && timings.sim_steps < ENTITY_SIM_STEP_MAX_PER_BROADCAST {
        collision_cache.clear();
        state.entity_store.simulate(*next_sim_ms, &state.content_registry, wasm_manager);
        let (mut step_explosions, mut step_player_modifiers) =
            simulate_mobs(state, &mut collision_cache, wasm_manager, *next_sim_ms, &mut timings);
        queued_explosions.append(&mut step_explosions);
        queued_player_modifiers.append(&mut step_player_modifiers);
        *next_sim_ms = (*next_sim_ms).saturating_add(sim_step_ms);
        timings.sim_steps += 1;
    }
    if timings.sim_steps == ENTITY_SIM_STEP_MAX_PER_BROADCAST && *next_sim_ms <= now_ms {
        *next_sim_ms = now_ms.saturating_add(sim_step_ms);
    }
    (queued_explosions, queued_player_modifiers, timings)
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::sync::Arc;
    use crate::content_registry::ContentRegistry;

    fn test_registry() -> Arc<ContentRegistry> {
        Arc::new(crate::plugin_loader::create_full_registry())
    }

    fn test_server_state_with_world() -> ServerState {
        let world = ServerWorldOverlay::from_chunk_payloads(
            crate::shared::voxel::BaseWorldKind::Empty,
            Vec::<([i32; 4], crate::shared::chunk_payload::ResolvedChunkPayload)>::new(),
            0,
            false,
            HashSet::new(),
        );
        ServerState::new(world, 1, false, false, Instant::now(), test_registry())
    }

    fn set_solid_voxel(state: &mut ServerState, x: i32, y: i32, z: i32, w: i32) {
        let _ = state.apply_world_voxel_edit([x, y, z, w], BlockData::simple(0, 1));
    }

    #[test]
    fn walking_neighbor_steps_include_elevation_transitions() {
        let steps = mob_nav_neighbor_steps(MobLocomotionMode::Walking);
        assert!(steps.contains(&[1, 1, 0, 0]));
        assert!(steps.contains(&[1, -1, 0, 0]));
        assert!(steps.contains(&[0, 1, 1, 0]));
        assert!(!steps.contains(&[0, 1, 0, 0]));
    }

    #[test]
    fn walking_line_of_sight_requires_supported_cells() {
        let mut state = test_server_state_with_world();
        set_solid_voxel(&mut state, 0, 1, 0, 0);
        set_solid_voxel(&mut state, 2, 1, 0, 0);

        let mut cache = HashMap::<ChunkKey, CollisionChunkCacheEntry>::new();
        let from = [0.0, 2.2, 0.0, 0.0];
        let to = [2.0, 2.2, 0.0, 0.0];
        assert!(!mob_nav_has_line_of_sight(
            &state,
            &mut cache,
            from,
            to,
            0.2,
            MobLocomotionMode::Walking
        ));

        set_solid_voxel(&mut state, 1, 1, 0, 0);
        cache.clear();
        assert!(mob_nav_has_line_of_sight(
            &state,
            &mut cache,
            from,
            to,
            0.2,
            MobLocomotionMode::Walking
        ));
    }

    #[test]
    fn walking_path_can_use_one_cell_step_up() {
        let mut state = test_server_state_with_world();
        set_solid_voxel(&mut state, 0, 1, 0, 0);
        set_solid_voxel(&mut state, 1, 2, 0, 0);
        set_solid_voxel(&mut state, 2, 2, 0, 0);

        let start = [0, 3, 0, 0];
        let goal = [2, 4, 0, 0];
        let mut cache = HashMap::<ChunkKey, CollisionChunkCacheEntry>::new();
        let path = mob_nav_find_path(&state, &mut cache, start, goal, 0.2, MobLocomotionMode::Walking)
            .expect("walking path should step up onto one-cell rise");
        assert!(path.reached_goal);
        assert_eq!(path.path_cells.last().copied(), Some(goal));

        let mut cursor = start;
        let mut saw_vertical_step = false;
        for cell in path.path_cells {
            if (cell[1] - cursor[1]).abs() == 1 {
                saw_vertical_step = true;
            }
            cursor = cell;
        }
        assert!(
            saw_vertical_step,
            "walking path should include an elevation transition"
        );
    }

    #[test]
    fn walking_step_resolution_climbs_lip_when_progress_is_blocked() {
        let mut state = test_server_state_with_world();
        set_solid_voxel(&mut state, 0, 0, 0, 0);
        set_solid_voxel(&mut state, 1, 0, 0, 0);
        set_solid_voxel(&mut state, 1, 1, 0, 0);
        set_solid_voxel(&mut state, 2, 1, 0, 0);

        let old_pos = [0.2, 2.0, 0.0, 0.0];
        let attempted_pos = [1.2, 2.0, 0.0, 0.0];
        let scale = 0.78;
        let mut cache = HashMap::<ChunkKey, CollisionChunkCacheEntry>::new();
        let baseline = resolve_mob_collision(&state, &mut cache, old_pos, attempted_pos, scale);
        let stepped = resolve_walking_collision_with_steps(&state, &mut cache, old_pos, attempted_pos, scale);
        assert!(
            stepped[0] > baseline[0] + 0.15,
            "expected step solver to improve horizontal progress: baseline_x={} stepped_x={}",
            baseline[0],
            stepped[0]
        );
    }

    #[test]
    fn ground_stick_respects_max_drop_budget() {
        let state = test_server_state_with_world();
        let start = [0.0, 10.0, 0.0, 0.0];
        let mut cache = HashMap::<ChunkKey, CollisionChunkCacheEntry>::new();
        let stuck = stick_mob_to_ground(&state, &mut cache, start, 0.3, MOB_WALK_GROUND_STICK_MAX_DROP);
        let dropped = start[1] - stuck[1];
        assert!(dropped <= MOB_WALK_GROUND_STICK_MAX_DROP + 1e-4);
        assert!(dropped + 1e-4 >= MOB_WALK_GROUND_STICK_MAX_DROP - MOB_WALK_GROUND_STICK_STEP);
    }
}
