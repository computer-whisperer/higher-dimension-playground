use super::super::types::MobNavCell;
use super::super::*;
use super::physics::{
    mob_collides_at, mob_collision_radius_for_scale, sample_effective_voxel_for_collision,
};
use crate::shared::region_tree::ChunkKey;

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

pub(super) fn mob_nav_position_from_cell(cell: MobNavCell) -> [f32; 4] {
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

pub(super) fn mob_nav_has_line_of_sight(
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
    let steps = (dist / MOB_NAV_PATH_LOS_STEP).ceil().clamp(1.0, 256.0) as usize;
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
            if !mob_nav_cell_is_walkable(state, cache, probe_cell, collision_radius, locomotion) {
                return false;
            }
        }
    }
    true
}

pub(super) fn mob_nav_cell_is_walkable(
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

pub(super) fn mob_nav_neighbor_steps(locomotion: MobLocomotionMode) -> &'static [MobNavCell] {
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

pub(super) fn mob_nav_find_walkable_goal_cell(
    state: &ServerState,
    cache: &mut HashMap<ChunkKey, CollisionChunkCacheEntry>,
    desired_goal: MobNavCell,
    origin: MobNavCell,
    collision_radius: f32,
    locomotion: MobLocomotionMode,
) -> Option<MobNavCell> {
    if mob_nav_cell_is_walkable(state, cache, desired_goal, collision_radius, locomotion) {
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

pub(super) fn mob_nav_find_path(
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

pub(super) fn update_mob_navigation_state(
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

    let has_los = if !goal_changed
        && now_ms.saturating_sub(navigation.last_los_check_ms) < MOB_NAV_LOS_CACHE_MS
    {
        navigation.last_los_result
    } else {
        let result = mob_nav_has_line_of_sight(
            state,
            cache,
            position,
            direct_target,
            collision_radius,
            locomotion,
        );
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

        if let Some(path_result) = mob_nav_find_path(
            state,
            cache,
            start_cell,
            goal_cell,
            collision_radius,
            locomotion,
        ) {
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
