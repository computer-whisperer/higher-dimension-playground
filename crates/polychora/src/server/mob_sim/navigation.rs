use super::super::types::MobNavCell;
use super::super::*;
use super::physics::{
    mob_collides_at, mob_collision_radius_for_scale, sample_effective_voxel_for_collision,
};
use crate::shared::region_tree::ChunkKey;
use std::sync::OnceLock;

/// Integer step costs so A* stays in integer math: a cardinal move costs 10
/// and a two-axis diagonal costs 14 (~= 10 * sqrt(2)).
pub(super) const MOB_NAV_CARDINAL_STEP_COST: i32 = 10;
pub(super) const MOB_NAV_DIAGONAL_STEP_COST: i32 = 14;

#[derive(Clone, Copy, Debug)]
pub(super) struct MobNavStep {
    pub(super) delta: MobNavCell,
    pub(super) cost: i32,
    /// For diagonal steps, the two axis-aligned deltas flanking the diagonal.
    /// Both flank cells must be walkable or the step is rejected: large mobs
    /// (collision radius > 0.5) sweep outside the endpoint cells mid-diagonal,
    /// and the waypoint follower cuts corners within the node reach distance.
    pub(super) corner_checks: Option<[MobNavCell; 2]>,
}

fn cardinal_step(axis: usize, sign: i32, dy: i32) -> MobNavStep {
    let mut delta = [0i32; 4];
    delta[axis] = sign;
    delta[1] += dy;
    MobNavStep {
        delta,
        cost: MOB_NAV_CARDINAL_STEP_COST,
        corner_checks: None,
    }
}

fn diagonal_step(axis_a: usize, sign_a: i32, axis_b: usize, sign_b: i32) -> MobNavStep {
    let mut delta = [0i32; 4];
    delta[axis_a] = sign_a;
    delta[axis_b] = sign_b;
    let mut flank_a = [0i32; 4];
    flank_a[axis_a] = sign_a;
    let mut flank_b = [0i32; 4];
    flank_b[axis_b] = sign_b;
    MobNavStep {
        delta,
        cost: MOB_NAV_DIAGONAL_STEP_COST,
        corner_checks: Some([flank_a, flank_b]),
    }
}

/// Walkers move along the three horizontal axes (x, z, w). Cardinal steps may
/// ride a one-cell elevation change; diagonals are flat only, so stairs and
/// lips are always negotiated on a single axis.
fn build_walking_steps() -> Vec<MobNavStep> {
    const HORIZONTAL_AXES: [usize; 3] = [0, 2, 3];
    let mut steps = Vec::with_capacity(30);
    for &axis in &HORIZONTAL_AXES {
        for sign in [1i32, -1] {
            for dy in [0i32, 1, -1] {
                steps.push(cardinal_step(axis, sign, dy));
            }
        }
    }
    for (idx, &axis_a) in HORIZONTAL_AXES.iter().enumerate() {
        for &axis_b in &HORIZONTAL_AXES[idx + 1..] {
            for sign_a in [1i32, -1] {
                for sign_b in [1i32, -1] {
                    steps.push(diagonal_step(axis_a, sign_a, axis_b, sign_b));
                }
            }
        }
    }
    steps
}

fn build_flying_steps() -> Vec<MobNavStep> {
    let mut steps = Vec::with_capacity(32);
    for axis in 0..4 {
        for sign in [1i32, -1] {
            steps.push(cardinal_step(axis, sign, 0));
        }
    }
    for axis_a in 0..4 {
        for axis_b in axis_a + 1..4 {
            for sign_a in [1i32, -1] {
                for sign_b in [1i32, -1] {
                    steps.push(diagonal_step(axis_a, sign_a, axis_b, sign_b));
                }
            }
        }
    }
    steps
}

pub(super) fn mob_nav_neighbor_steps(locomotion: MobLocomotionMode) -> &'static [MobNavStep] {
    static WALKING_STEPS: OnceLock<Vec<MobNavStep>> = OnceLock::new();
    static FLYING_STEPS: OnceLock<Vec<MobNavStep>> = OnceLock::new();
    match locomotion {
        MobLocomotionMode::Walking => WALKING_STEPS.get_or_init(build_walking_steps),
        MobLocomotionMode::Flying => FLYING_STEPS.get_or_init(build_flying_steps),
    }
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

/// Exact obstacle-free path cost over the given per-axis deltas using
/// cardinal and two-axis diagonal steps: pairing two units from distinct
/// axes into one diagonal (14) beats two cardinals (20), and the number of
/// such pairings is limited by the dominant axis.
fn mob_nav_pairing_cost(axis_deltas: &[i32]) -> i32 {
    let mut sum = 0i32;
    let mut max_axis = 0i32;
    for &delta in axis_deltas {
        sum = sum.saturating_add(delta);
        max_axis = max_axis.max(delta);
    }
    let pairs = (sum - max_axis).min(sum / 2);
    (MOB_NAV_CARDINAL_STEP_COST.saturating_mul(sum))
        .saturating_sub((2 * MOB_NAV_CARDINAL_STEP_COST - MOB_NAV_DIAGONAL_STEP_COST) * pairs)
}

/// Admissible, consistent heuristic for the step sets above (exact on open
/// terrain). Walkers pay nothing extra for elevation riding along a cardinal
/// step, but every step changes y by at most one cell, so the vertical term
/// is still a valid lower bound.
fn mob_nav_heuristic_cost(a: MobNavCell, b: MobNavCell, locomotion: MobLocomotionMode) -> i32 {
    let dx = a[0].abs_diff(b[0]) as i32;
    let dy = a[1].abs_diff(b[1]) as i32;
    let dz = a[2].abs_diff(b[2]) as i32;
    let dw = a[3].abs_diff(b[3]) as i32;
    match locomotion {
        MobLocomotionMode::Walking => mob_nav_pairing_cost(&[dx, dz, dw])
            .max(MOB_NAV_CARDINAL_STEP_COST.saturating_mul(dy)),
        MobLocomotionMode::Flying => mob_nav_pairing_cost(&[dx, dy, dz, dw]),
    }
}

/// Per-search cache of cell walkability. During one navigation update the
/// world, the mob's collision radius, and its locomotion mode are all fixed,
/// so a cell's verdict can be reused across LOS probes, goal snapping, and
/// every A* expansion that touches it (a cell has up to 30 in-neighbors).
#[derive(Default)]
pub(super) struct MobNavWalkabilityMemo {
    cells: HashMap<MobNavCell, bool>,
}

pub(super) fn mob_nav_has_line_of_sight(
    state: &ServerState,
    cache: &mut HashMap<ChunkKey, CollisionChunkCacheEntry>,
    memo: &mut MobNavWalkabilityMemo,
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
            if !mob_nav_cell_is_walkable(
                state,
                cache,
                memo,
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

pub(super) fn mob_nav_cell_is_walkable(
    state: &ServerState,
    cache: &mut HashMap<ChunkKey, CollisionChunkCacheEntry>,
    memo: &mut MobNavWalkabilityMemo,
    cell: MobNavCell,
    collision_radius: f32,
    locomotion: MobLocomotionMode,
) -> bool {
    if let Some(&walkable) = memo.cells.get(&cell) {
        return walkable;
    }
    let mut walkable = !mob_collides_at(
        state,
        cache,
        mob_nav_position_from_cell(cell),
        collision_radius,
    );
    if walkable && locomotion == MobLocomotionMode::Walking {
        walkable = sample_effective_voxel_for_collision(
            state,
            cache,
            cell[0],
            cell[1] - 2,
            cell[2],
            cell[3],
        );
    }
    memo.cells.insert(cell, walkable);
    walkable
}

/// L1 shells around a cell, nearest first, out to the goal-adjust radius.
fn mob_nav_goal_adjust_rings() -> &'static [Vec<MobNavCell>] {
    static RINGS: OnceLock<Vec<Vec<MobNavCell>>> = OnceLock::new();
    RINGS.get_or_init(|| {
        let max_radius = MOB_NAV_PATH_GOAL_ADJUST_RADIUS_CELLS.max(0);
        let mut rings = vec![Vec::new(); max_radius as usize];
        for dx in -max_radius..=max_radius {
            for dy in -max_radius..=max_radius {
                for dz in -max_radius..=max_radius {
                    for dw in -max_radius..=max_radius {
                        let ring = dx.abs() + dy.abs() + dz.abs() + dw.abs();
                        if ring == 0 || ring > max_radius {
                            continue;
                        }
                        rings[(ring - 1) as usize].push([dx, dy, dz, dw]);
                    }
                }
            }
        }
        rings
    })
}

pub(super) fn mob_nav_find_walkable_goal_cell(
    state: &ServerState,
    cache: &mut HashMap<ChunkKey, CollisionChunkCacheEntry>,
    memo: &mut MobNavWalkabilityMemo,
    desired_goal: MobNavCell,
    origin: MobNavCell,
    collision_radius: f32,
    locomotion: MobLocomotionMode,
) -> Option<MobNavCell> {
    if mob_nav_cell_is_walkable(state, cache, memo, desired_goal, collision_radius, locomotion) {
        return Some(desired_goal);
    }

    // Scan shells nearest-first and stop at the first one holding a walkable
    // cell, breaking ties within a shell toward the origin. A farther shell
    // can never win: candidates within the adjust radius differ in origin
    // distance by at most twice the radius, less than one shell's weight.
    for ring_offsets in mob_nav_goal_adjust_rings() {
        let mut best: Option<(i32, MobNavCell)> = None;
        for offset in ring_offsets {
            let candidate = [
                desired_goal[0] + offset[0],
                desired_goal[1] + offset[1],
                desired_goal[2] + offset[2],
                desired_goal[3] + offset[3],
            ];
            if !mob_nav_cell_is_walkable(
                state,
                cache,
                memo,
                candidate,
                collision_radius,
                locomotion,
            ) {
                continue;
            }
            let origin_distance = mob_nav_manhattan_distance(candidate, origin);
            if best.map_or(true, |(dist, _)| origin_distance < dist) {
                best = Some((origin_distance, candidate));
            }
        }
        if let Some((_, cell)) = best {
            return Some(cell);
        }
    }
    None
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
    memo: &mut MobNavWalkabilityMemo,
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
    if !mob_nav_cell_is_walkable(state, cache, memo, start, collision_radius, locomotion) {
        return None;
    }
    if !mob_nav_cell_is_walkable(state, cache, memo, goal, collision_radius, locomotion) {
        return None;
    }

    let mut open = BinaryHeap::<(Reverse<i32>, Reverse<i32>, MobNavCell)>::new();
    let mut g_scores = HashMap::<MobNavCell, i32>::new();
    let mut came_from = HashMap::<MobNavCell, MobNavCell>::new();
    let mut best_cell = start;
    let mut best_h = mob_nav_heuristic_cost(start, goal, locomotion);
    let mut best_g = 0i32;

    g_scores.insert(start, 0);
    open.push((Reverse(best_h), Reverse(0), start));

    let mut visited_steps = 0usize;
    while let Some((_f_score, Reverse(g_cost), cell)) = open.pop() {
        let current_best = g_scores.get(&cell).copied().unwrap_or(i32::MAX);
        if g_cost > current_best {
            continue;
        }
        let h_cost = mob_nav_heuristic_cost(cell, goal, locomotion);
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
                cell[0] + step.delta[0],
                cell[1] + step.delta[1],
                cell[2] + step.delta[2],
                cell[3] + step.delta[3],
            ];
            if mob_nav_manhattan_distance(start, next) > MOB_NAV_PATH_MAX_SEARCH_RADIUS_CELLS {
                continue;
            }

            let tentative_g = g_cost.saturating_add(step.cost);
            let known_next_g = g_scores.get(&next).copied().unwrap_or(i32::MAX);
            if tentative_g >= known_next_g {
                continue;
            }
            if !mob_nav_cell_is_walkable(state, cache, memo, next, collision_radius, locomotion) {
                continue;
            }
            if let Some(corners) = step.corner_checks {
                let corner_blocked = corners.iter().any(|flank| {
                    let flank_cell = [
                        cell[0] + flank[0],
                        cell[1] + flank[1],
                        cell[2] + flank[2],
                        cell[3] + flank[3],
                    ];
                    !mob_nav_cell_is_walkable(
                        state,
                        cache,
                        memo,
                        flank_cell,
                        collision_radius,
                        locomotion,
                    )
                });
                if corner_blocked {
                    continue;
                }
            }

            came_from.insert(next, cell);
            g_scores.insert(next, tentative_g);
            let h_cost = mob_nav_heuristic_cost(next, goal, locomotion);
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

#[allow(clippy::too_many_arguments)]
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
    let mut memo = MobNavWalkabilityMemo::default();
    let direct_target = match locomotion {
        MobLocomotionMode::Walking => [target[0], position[1], target[2], target[3]],
        MobLocomotionMode::Flying => target,
    };
    let desired_start_cell = mob_nav_base_cell_from_position(position, locomotion);
    let start_cell = mob_nav_find_walkable_goal_cell(
        state,
        cache,
        &mut memo,
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
            &mut memo,
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
            &mut memo,
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
            &mut memo,
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
