use super::runtime_net::{apply_creeper_explosion, apply_explosion_impulse};
use super::*;

fn mob_collision_radius_for_scale(scale: f32) -> f32 {
    let clamped_scale = if scale.is_finite() { scale } else { 0.5 };
    (clamped_scale * MOB_COLLISION_RADIUS_SCALE)
        .clamp(MOB_COLLISION_RADIUS_MIN, MOB_COLLISION_RADIUS_MAX)
}

fn sample_effective_voxel_for_collision(
    state: &ServerState,
    cache: &mut HashMap<ChunkPos, CollisionChunkCacheEntry>,
    wx: i32,
    wy: i32,
    wz: i32,
    ww: i32,
) -> VoxelType {
    let (chunk_pos, voxel_index) = voxel::world_to_chunk(wx, wy, wz, ww);
    if let Some(entry) = cache.get(&chunk_pos) {
        return match entry {
            CollisionChunkCacheEntry::Explicit(chunk) => chunk[voxel_index],
            CollisionChunkCacheEntry::ExplicitEmpty => VoxelType::AIR,
            CollisionChunkCacheEntry::Effective(chunk) => chunk
                .as_ref()
                .map(|chunk| chunk[voxel_index])
                .unwrap_or(VoxelType::AIR),
        };
    }

    if let Some(chunk_data) = state.world_chunk_at(chunk_pos) {
        if chunk_data.iter().all(|voxel| voxel.is_air()) {
            cache.insert(chunk_pos, CollisionChunkCacheEntry::ExplicitEmpty);
            return VoxelType::AIR;
        }
        let voxel = chunk_data[voxel_index];
        cache.insert(chunk_pos, CollisionChunkCacheEntry::Explicit(chunk_data));
        return voxel;
    }

    let effective_chunk = state.world_effective_chunk(chunk_pos, false);
    let voxel = effective_chunk
        .as_ref()
        .map(|chunk| chunk[voxel_index])
        .unwrap_or(VoxelType::AIR);
    cache.insert(
        chunk_pos,
        CollisionChunkCacheEntry::Effective(effective_chunk),
    );
    voxel
}

fn mob_collides_at(
    state: &ServerState,
    cache: &mut HashMap<ChunkPos, CollisionChunkCacheEntry>,
    position: [f32; 4],
    radius: f32,
) -> bool {
    if position[1] - radius < MOB_HARD_WORLD_FLOOR_Y {
        return true;
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
                    if sample_effective_voxel_for_collision(state, cache, x, y, z, w).is_solid() {
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
    old_pos: [f32; 4],
    attempted_pos: [f32; 4],
    scale: f32,
) -> [f32; 4] {
    let radius = mob_collision_radius_for_scale(scale);
    let mut cache = HashMap::<ChunkPos, CollisionChunkCacheEntry>::new();
    let mut pos = old_pos;
    if mob_collides_at(state, &mut cache, pos, radius) {
        for _ in 0..MOB_COLLISION_MAX_PUSHUP_STEPS {
            pos[1] += MOB_COLLISION_PUSHUP_STEP;
            if !mob_collides_at(state, &mut cache, pos, radius) {
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
        if !mob_collides_at(state, &mut cache, candidate, radius) {
            pos = candidate;
            continue;
        }

        let mut feasible = pos[axis];
        let mut blocked = target;
        for _ in 0..MOB_COLLISION_BINARY_STEPS {
            let mid = 0.5 * (feasible + blocked);
            let mut probe = pos;
            probe[axis] = mid;
            if mob_collides_at(state, &mut cache, probe, radius) {
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

fn mob_nav_base_cell_from_position(position: [f32; 4]) -> MobNavCell {
    [
        position[0].round() as i32,
        position[1].ceil() as i32,
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
    from: [f32; 4],
    to: [f32; 4],
    collision_radius: f32,
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
    let mut cache = HashMap::<ChunkPos, CollisionChunkCacheEntry>::new();
    for idx in 1..=steps {
        let t = idx as f32 / steps as f32;
        let probe = [
            from[0] + delta[0] * t,
            from[1] + delta[1] * t,
            from[2] + delta[2] * t,
            from[3] + delta[3] * t,
        ];
        if mob_collides_at(state, &mut cache, probe, collision_radius) {
            return false;
        }
    }
    true
}

fn mob_nav_find_walkable_goal_cell(
    state: &ServerState,
    desired_goal: MobNavCell,
    origin: MobNavCell,
    collision_radius: f32,
) -> Option<MobNavCell> {
    let mut cache = HashMap::<ChunkPos, CollisionChunkCacheEntry>::new();
    if !mob_collides_at(
        state,
        &mut cache,
        mob_nav_position_from_cell(desired_goal),
        collision_radius,
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
                    if mob_collides_at(
                        state,
                        &mut cache,
                        mob_nav_position_from_cell(candidate),
                        collision_radius,
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
    start: MobNavCell,
    goal: MobNavCell,
    collision_radius: f32,
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
    let mut cache = HashMap::<ChunkPos, CollisionChunkCacheEntry>::new();
    if mob_collides_at(
        state,
        &mut cache,
        mob_nav_position_from_cell(start),
        collision_radius,
    ) {
        return None;
    }
    if mob_collides_at(
        state,
        &mut cache,
        mob_nav_position_from_cell(goal),
        collision_radius,
    ) {
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

        for step in [
            [1, 0, 0, 0],
            [-1, 0, 0, 0],
            [0, 1, 0, 0],
            [0, -1, 0, 0],
            [0, 0, 1, 0],
            [0, 0, -1, 0],
            [0, 0, 0, 1],
            [0, 0, 0, -1],
        ] {
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
            if mob_collides_at(
                state,
                &mut cache,
                mob_nav_position_from_cell(next),
                collision_radius,
            ) {
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
    archetype: MobArchetype,
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
        "[mob-nav][server] t={} entity={} archetype={:?} {}",
        now_ms, mob_entity_id, archetype, message
    );
}

fn update_mob_navigation_state(
    state: &ServerState,
    mut navigation: MobNavigationState,
    mob_entity_id: u64,
    archetype: MobArchetype,
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
        return (None, false, navigation);
    };

    let collision_radius = mob_collision_radius_for_scale(scale);
    let desired_start_cell = mob_nav_base_cell_from_position(position);
    let start_cell = mob_nav_find_walkable_goal_cell(
        state,
        desired_start_cell,
        desired_start_cell,
        collision_radius,
    )
    .unwrap_or(desired_start_cell);
    let desired_goal_cell = mob_nav_base_cell_from_position(target);
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

    let has_los = mob_nav_has_line_of_sight(state, position, target, collision_radius);
    if has_los {
        navigation.goal_cell = Some(desired_goal_cell);
        navigation.path_cells.clear();
        navigation.path_cursor = 0;
        if goal_changed || path_exhausted || repath_due {
            mob_nav_debug_log(
                &mut navigation,
                debug_enabled,
                now_ms,
                mob_entity_id,
                archetype,
                &format!(
                    "mode=los start={:?} goal={:?} path_exhausted={} repath_due={}",
                    start_cell, desired_goal_cell, path_exhausted, repath_due
                ),
            );
        }
        return (Some(target), false, navigation);
    }

    if goal_changed || path_exhausted || repath_due {
        let goal_cell =
            mob_nav_find_walkable_goal_cell(state, desired_goal_cell, start_cell, collision_radius)
                .unwrap_or(desired_goal_cell);
        navigation.goal_cell = Some(goal_cell);
        navigation.last_repath_ms = now_ms;

        if let Some(path_result) = mob_nav_find_path(state, start_cell, goal_cell, collision_radius)
        {
            let path_len = path_result.path_cells.len();
            let reached_goal = path_result.reached_goal;
            let expanded_steps = path_result.expanded_steps;
            let best_goal_distance = path_result.best_goal_distance;
            let best_cell = path_result.best_cell;
            navigation.path_cells = path_result.path_cells;
            navigation.path_cursor = 0;
            mob_nav_debug_log(
                &mut navigation,
                debug_enabled,
                now_ms,
                mob_entity_id,
                archetype,
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
            mob_nav_debug_log(
                &mut navigation,
                debug_enabled,
                now_ms,
                mob_entity_id,
                archetype,
                &format!(
                    "mode=path-fail start={:?} goal={:?} (no reachable cell found)",
                    start_cell, goal_cell
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
    } else {
        (Some(target), false, navigation)
    }
}

fn simulate_seeker_step(
    mob: &MobState,
    position: [f32; 4],
    target_position: Option<[f32; 4]>,
    path_following: bool,
    simple_steer: bool,
    now_ms: u64,
    previous_update_ms: u64,
) -> ([f32; 4], [f32; 4]) {
    let dt_s = (now_ms.saturating_sub(previous_update_ms)).max(1) as f32 * 0.001;
    let t_s = now_ms as f32 * 0.001;
    let (desired_dir, slow_factor) = if let Some(target) = target_position {
        let to_target = [
            target[0] - position[0],
            target[1] - position[1],
            target[2] - position[2],
            target[3] - position[3],
        ];
        let distance = distance4_sq(target, position).sqrt();
        let direct = normalize4_or_default(to_target, [0.0, 0.0, 1.0, 0.0]);
        if path_following {
            let slow = if distance < MOB_NAV_PATH_NODE_REACH_DISTANCE * 0.9 {
                0.62
            } else {
                1.05
            };
            (direct, slow)
        } else if simple_steer {
            let slow = if distance < mob.preferred_distance * 0.6 {
                0.42
            } else {
                1.0
            };
            (direct, slow)
        } else {
            // A 4D tangent that rotates [x,z] and [y,w] planes together.
            let tangent = normalize4_or_default(
                [-direct[2], -direct[3], direct[0], direct[1]],
                [direct[3], 0.0, -direct[0], direct[1]],
            );
            let weave_phase = t_s * 1.7 + mob.phase_offset;
            let weave = [
                0.10 * weave_phase.sin(),
                0.08 * (weave_phase * 0.7).cos(),
                -0.10 * weave_phase.sin(),
                0.08 * (weave_phase * 1.3).sin(),
            ];
            let pursuit = if distance > mob.preferred_distance {
                1.0
            } else {
                -0.45
            };
            let slow = if distance < mob.preferred_distance * 0.6 {
                0.42
            } else {
                1.0
            };
            let desired = normalize4_or_default(
                [
                    direct[0] * pursuit + tangent[0] * mob.tangent_weight + weave[0],
                    direct[1] * pursuit + tangent[1] * mob.tangent_weight + weave[1],
                    direct[2] * pursuit + tangent[2] * mob.tangent_weight + weave[2],
                    direct[3] * pursuit + tangent[3] * mob.tangent_weight + weave[3],
                ],
                direct,
            );
            (desired, slow)
        }
    } else {
        let phase = t_s * 0.65 + mob.phase_offset;
        (
            normalize4_or_default(
                [
                    phase.cos(),
                    0.45 * (phase * 0.7).sin(),
                    phase.sin(),
                    (phase * 1.1).cos(),
                ],
                [0.0, 0.0, 1.0, 0.0],
            ),
            0.35,
        )
    };

    let step = mob.move_speed.max(0.1) * slow_factor * dt_s;
    let next_position = [
        position[0] + desired_dir[0] * step,
        position[1] + desired_dir[1] * step,
        position[2] + desired_dir[2] * step,
        position[3] + desired_dir[3] * step,
    ];
    (next_position, desired_dir)
}

fn simulate_creeper_step(
    mob: &MobState,
    position: [f32; 4],
    target_position: Option<[f32; 4]>,
    path_following: bool,
    simple_steer: bool,
    now_ms: u64,
    previous_update_ms: u64,
) -> ([f32; 4], [f32; 4]) {
    let dt_s = (now_ms.saturating_sub(previous_update_ms)).max(1) as f32 * 0.001;
    let t_s = now_ms as f32 * 0.001;
    let (desired_dir, speed_factor) = if let Some(target) = target_position {
        let to_target = [
            target[0] - position[0],
            target[1] - position[1],
            target[2] - position[2],
            target[3] - position[3],
        ];
        let distance = distance4_sq(target, position).sqrt();
        let direct = normalize4_or_default(to_target, [0.0, 0.0, 1.0, 0.0]);
        if path_following {
            let speed = if distance > MOB_NAV_PATH_NODE_REACH_DISTANCE * 1.2 {
                1.18
            } else {
                0.66
            };
            (direct, speed)
        } else if simple_steer {
            let speed = if distance > mob.preferred_distance * 0.9 {
                1.15
            } else {
                0.70
            };
            (direct, speed)
        } else {
            // Creeper keeps pressure by orbiting in two coupled 4D planes, then lunges.
            let orbit_a = normalize4_or_default(
                [-direct[2], direct[3], direct[0], -direct[1]],
                [direct[3], 0.0, -direct[0], direct[1]],
            );
            let orbit_b = normalize4_or_default(
                [direct[1], -direct[0], direct[3], -direct[2]],
                [0.0, direct[2], -direct[1], direct[0]],
            );
            let phase = t_s * 2.35 + mob.phase_offset;
            let orbit_mix = [
                orbit_a[0] * phase.sin() + orbit_b[0] * phase.cos(),
                orbit_a[1] * phase.sin() + orbit_b[1] * phase.cos(),
                orbit_a[2] * phase.sin() + orbit_b[2] * phase.cos(),
                orbit_a[3] * phase.sin() + orbit_b[3] * phase.cos(),
            ];
            let too_close = distance < mob.preferred_distance * 0.55;
            let in_lunge_band = distance > mob.preferred_distance * 0.80
                && distance < mob.preferred_distance * 1.95;
            let lunge = in_lunge_band && phase.sin() > 0.58;
            let pressure = if too_close {
                -0.72
            } else if lunge {
                1.85
            } else {
                0.68
            };
            let speed = if lunge {
                1.65
            } else if too_close {
                0.48
            } else {
                0.95
            };

            let desired = normalize4_or_default(
                [
                    direct[0] * pressure + orbit_mix[0] * mob.tangent_weight,
                    direct[1] * pressure + orbit_mix[1] * mob.tangent_weight,
                    direct[2] * pressure + orbit_mix[2] * mob.tangent_weight,
                    direct[3] * pressure + orbit_mix[3] * mob.tangent_weight,
                ],
                direct,
            );
            (desired, speed)
        }
    } else {
        let phase = t_s * 0.72 + mob.phase_offset;
        (
            normalize4_or_default(
                [
                    0.7 * phase.sin(),
                    (phase * 0.9).cos(),
                    0.7 * phase.cos(),
                    (phase * 1.4).sin(),
                ],
                [0.0, 0.0, 1.0, 0.0],
            ),
            0.38,
        )
    };

    let step = mob.move_speed.max(0.1) * speed_factor * dt_s;
    let next_position = [
        position[0] + desired_dir[0] * step,
        position[1] + desired_dir[1] * step,
        position[2] + desired_dir[2] * step,
        position[3] + desired_dir[3] * step,
    ];
    (next_position, desired_dir)
}

fn simulate_phase_spider_step(
    mob: &MobState,
    position: [f32; 4],
    target_position: Option<[f32; 4]>,
    path_following: bool,
    simple_steer: bool,
    now_ms: u64,
    previous_update_ms: u64,
) -> ([f32; 4], [f32; 4]) {
    let dt_s = (now_ms.saturating_sub(previous_update_ms)).max(1) as f32 * 0.001;
    let t_s = now_ms as f32 * 0.001;
    let (desired_dir, speed_factor) = if let Some(target) = target_position {
        let to_target = [
            target[0] - position[0],
            target[1] - position[1],
            target[2] - position[2],
            target[3] - position[3],
        ];
        let distance = distance4_sq(target, position).sqrt();
        let direct = normalize4_or_default(to_target, [0.0, 0.0, 1.0, 0.0]);
        if path_following {
            let phase = t_s * 6.8 + mob.phase_offset;
            let tangent = normalize4_or_default(
                [direct[3], 0.0, -direct[0], direct[2]],
                [-direct[2], direct[3], direct[0], -direct[1]],
            );
            let lateral_scale = if simple_steer { 0.0 } else { 0.18 };
            let desired = normalize4_or_default(
                [
                    direct[0] + tangent[0] * lateral_scale * phase.sin(),
                    direct[1] + tangent[1] * lateral_scale * (phase * 0.7).sin(),
                    direct[2] + tangent[2] * lateral_scale * phase.sin(),
                    direct[3] + tangent[3] * lateral_scale * (phase * 0.9).cos(),
                ],
                direct,
            );
            let speed = if distance > MOB_NAV_PATH_NODE_REACH_DISTANCE * 1.4 {
                1.32
            } else {
                0.78
            };
            (desired, speed)
        } else if simple_steer {
            let speed = if distance > mob.preferred_distance {
                1.20
            } else {
                0.74
            };
            (direct, speed)
        } else {
            let phase = t_s * 5.2 + mob.phase_offset;
            let strafe_a = normalize4_or_default(
                [-direct[2], direct[3], direct[0], -direct[1]],
                [direct[3], 0.0, -direct[0], direct[1]],
            );
            let strafe_b = normalize4_or_default(
                [direct[1], -direct[0], direct[3], -direct[2]],
                [0.0, direct[2], -direct[1], direct[0]],
            );
            let strafe_mix = [
                strafe_a[0] * phase.sin() + strafe_b[0] * phase.cos(),
                strafe_a[1] * phase.sin() + strafe_b[1] * phase.cos(),
                strafe_a[2] * phase.sin() + strafe_b[2] * phase.cos(),
                strafe_a[3] * phase.sin() + strafe_b[3] * phase.cos(),
            ];
            let stalk = if distance > mob.preferred_distance * 1.35 {
                1.0
            } else if distance < mob.preferred_distance * 0.65 {
                -0.52
            } else {
                0.24
            };
            let vertical = 0.11 * (phase * 1.4).sin();
            let desired = normalize4_or_default(
                [
                    direct[0] * stalk + strafe_mix[0] * mob.tangent_weight * 1.30,
                    direct[1] * stalk + strafe_mix[1] * mob.tangent_weight * 1.30 + vertical,
                    direct[2] * stalk + strafe_mix[2] * mob.tangent_weight * 1.30,
                    direct[3] * stalk + strafe_mix[3] * mob.tangent_weight * 1.30,
                ],
                direct,
            );
            let speed = if distance > mob.preferred_distance {
                1.15
            } else {
                0.70
            };
            (desired, speed)
        }
    } else {
        let phase = t_s * 0.93 + mob.phase_offset;
        (
            normalize4_or_default(
                [
                    0.9 * phase.sin(),
                    0.28 * (phase * 1.1).cos(),
                    0.9 * phase.cos(),
                    (phase * 1.6).sin(),
                ],
                [0.0, 0.0, 1.0, 0.0],
            ),
            0.44,
        )
    };

    let step = mob.move_speed.max(0.1) * speed_factor * dt_s;
    let next_position = [
        position[0] + desired_dir[0] * step,
        position[1] + desired_dir[1] * step,
        position[2] + desired_dir[2] * step,
        position[3] + desired_dir[3] * step,
    ];
    (next_position, desired_dir)
}

fn attempt_phase_spider_blink(
    state: &ServerState,
    position: [f32; 4],
    forward: [f32; 4],
    scale: f32,
    phase_offset: f32,
    now_ms: u64,
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
    let mut cache = HashMap::<ChunkPos, CollisionChunkCacheEntry>::new();
    let lift_primary = 0.16 + 0.12 * (phase * 0.8).sin().abs();
    let lift_options = [lift_primary, 0.05, -0.08];

    let mut distance = PHASE_SPIDER_PHASE_DISTANCE;
    while distance >= PHASE_SPIDER_PHASE_MIN_DISTANCE {
        for &lift in &lift_options {
            let candidate = [
                position[0] + blink_dir[0] * distance,
                position[1] + blink_dir[1] * distance + lift,
                position[2] + blink_dir[2] * distance,
                position[3] + blink_dir[3] * distance,
            ];
            if !mob_collides_at(state, &mut cache, candidate, radius) {
                return Some(candidate);
            }
        }
        distance -= 0.55;
    }
    None
}

fn simulate_mobs(
    state: &mut ServerState,
    now_ms: u64,
) -> (Vec<QueuedExplosionEvent>, Vec<QueuedPlayerMovementModifier>) {
    let player_positions: Vec<[f32; 4]> = state
        .players
        .values()
        .filter_map(|player| state.entity_store.snapshot(player.entity_id))
        .map(|snapshot| snapshot.position)
        .collect();

    let mut stale = Vec::new();
    let mut detonations = Vec::new();
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
        let nearest_target = nearest_position_to(snapshot.position, &player_positions);
        let navigation_target = match mob.archetype {
            MobArchetype::Seeker => nearest_target,
            MobArchetype::Creeper4d => nearest_target.map(|target| {
                [
                    target[0],
                    target[1] - CREEPER_POUNCE_TARGET_BELOW_PLAYER_Y,
                    target[2],
                    target[3],
                ]
            }),
            MobArchetype::PhaseSpider => nearest_target,
        };
        if mob.archetype == MobArchetype::Creeper4d {
            let should_detonate = player_positions.iter().any(|player_position| {
                distance4_sq(snapshot.position, *player_position).sqrt()
                    <= CREEPER_EXPLOSION_TRIGGER_DISTANCE
            });
            if should_detonate {
                detonations.push((mob.entity_id, snapshot.position));
                continue;
            }
        }
        let (target_position, path_following, mut navigation) = update_mob_navigation_state(
            state,
            mob.navigation.clone(),
            mob.entity_id,
            mob.archetype,
            state.mob_nav_debug,
            snapshot.position,
            snapshot.scale,
            navigation_target,
            now_ms,
        );
        let (next_position, next_forward) = match mob.archetype {
            MobArchetype::Seeker => simulate_seeker_step(
                &mob,
                snapshot.position,
                target_position,
                path_following,
                state.mob_nav_simple_steer,
                now_ms,
                snapshot.last_update_ms,
            ),
            MobArchetype::Creeper4d => simulate_creeper_step(
                &mob,
                snapshot.position,
                target_position,
                path_following,
                state.mob_nav_simple_steer,
                now_ms,
                snapshot.last_update_ms,
            ),
            MobArchetype::PhaseSpider => simulate_phase_spider_step(
                &mob,
                snapshot.position,
                target_position,
                path_following,
                state.mob_nav_simple_steer,
                now_ms,
                snapshot.last_update_ms,
            ),
        };
        let mut final_position =
            resolve_mob_collision(state, snapshot.position, next_position, snapshot.scale);
        let mut final_forward = next_forward;
        if mob.archetype == MobArchetype::PhaseSpider && now_ms >= mob.next_phase_ms {
            let attempted = distance4_sq(snapshot.position, next_position).sqrt();
            let resolved = distance4_sq(snapshot.position, final_position).sqrt();
            let blocked =
                attempted > 0.12 && resolved + PHASE_SPIDER_BLOCKED_PROGRESS_EPSILON < attempted;
            let should_phase = nearest_target.is_some() && (path_following || blocked);
            if should_phase {
                if let Some(phase_position) = attempt_phase_spider_blink(
                    state,
                    snapshot.position,
                    next_forward,
                    snapshot.scale,
                    mob.phase_offset,
                    now_ms,
                ) {
                    final_forward = normalize4_or_default(
                        [
                            phase_position[0] - snapshot.position[0],
                            phase_position[1] - snapshot.position[1],
                            phase_position[2] - snapshot.position[2],
                            phase_position[3] - snapshot.position[3],
                        ],
                        next_forward,
                    );
                    final_position = phase_position;
                    mob_nav_debug_log(
                        &mut navigation,
                        state.mob_nav_debug,
                        now_ms,
                        mob.entity_id,
                        mob.archetype,
                        &format!("mode=phase-blink success pos={:?}", phase_position),
                    );
                } else {
                    mob_nav_debug_log(
                        &mut navigation,
                        state.mob_nav_debug,
                        now_ms,
                        mob.entity_id,
                        mob.archetype,
                        "mode=phase-blink fail (no valid destination)",
                    );
                }
            }
            let next_deadline = phase_spider_next_phase_deadline(
                now_ms,
                mob.phase_offset,
                (mob.entity_id as u32) ^ (now_ms as u32),
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
    for (entity_id, center) in detonations {
        if !state.mobs.contains_key(&entity_id) {
            continue;
        }
        let (changed_chunk_count, explosion) =
            apply_creeper_explosion(state, entity_id, center, CREEPER_EXPLOSION_RADIUS_VOXELS);
        queued_explosions.push(explosion);
        let (_persistent_motion, mut player_modifiers) = apply_explosion_impulse(
            state,
            entity_id,
            center,
            CREEPER_EXPLOSION_IMPULSE_RADIUS,
            CREEPER_EXPLOSION_MAX_IMPULSE_DISTANCE,
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
    now_ms: u64,
    next_sim_ms: &mut u64,
    sim_step_ms: u64,
) -> (Vec<QueuedExplosionEvent>, Vec<QueuedPlayerMovementModifier>) {
    let mut queued_explosions = Vec::new();
    let mut queued_player_modifiers = Vec::new();
    let mut sim_steps = 0usize;
    while *next_sim_ms <= now_ms && sim_steps < ENTITY_SIM_STEP_MAX_PER_BROADCAST {
        let moved_entities = state.entity_store.simulate(*next_sim_ms);
        let _persistent_accent_motion = moved_entities.iter().any(|entity_id| {
            state
                .entity_records
                .get(entity_id)
                .map(|record| {
                    record.lifecycle == EntityLifecycle::Live
                        && record.persistent
                        && record.class == EntityClass::Accent
                })
                .unwrap_or(false)
        });
        let (mut step_explosions, mut step_player_modifiers) = simulate_mobs(state, *next_sim_ms);
        queued_explosions.append(&mut step_explosions);
        queued_player_modifiers.append(&mut step_player_modifiers);
        *next_sim_ms = (*next_sim_ms).saturating_add(sim_step_ms);
        sim_steps += 1;
    }
    if sim_steps == ENTITY_SIM_STEP_MAX_PER_BROADCAST && *next_sim_ms <= now_ms {
        *next_sim_ms = now_ms.saturating_add(sim_step_ms);
    }
    (queued_explosions, queued_player_modifiers)
}
