mod navigation;
mod physics;

use super::runtime_net::{apply_creeper_explosion, apply_explosion_impulse};
use super::*;
use crate::shared::region_tree::ChunkKey;
use crate::shared::wasm::{WasmPluginManager, WasmPluginSlot};
use navigation::update_mob_navigation_state;
use physics::{
    apply_mob_locomotion_post_collision, integrate_mob_steering_step, mob_collides_at,
    mob_collision_radius_for_scale,
};
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
        .call_slot(
            WasmPluginSlot::EntitySimulation,
            OP_ENTITY_TICK as i32,
            &input_bytes,
        )
        .ok()??;
    let output: EntityTickOutput = postcard::from_bytes(&result.invocation.output).ok()?;
    match output {
        EntityTickOutput::Steer {
            desired_direction,
            speed_factor,
        } => {
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

#[derive(Clone, Copy, Debug)]
struct MobSteeringCommand {
    desired_direction: [f32; 4],
    speed_factor: f32,
}

fn nearest_position_to(position: [f32; 4], candidates: &[[f32; 4]]) -> Option<[f32; 4]> {
    candidates.iter().copied().min_by(|a, b| {
        distance4_sq(*a, position)
            .partial_cmp(&distance4_sq(*b, position))
            .unwrap_or(std::cmp::Ordering::Equal)
    })
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
    let mut detonations: Vec<(u64, [f32; 4], polychora_plugin_api::entity::EntitySimConfig)> =
        Vec::new();
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
        let Some(sim_config) = state
            .content_registry
            .sim_config(mob.entity_ns, mob.entity_type)
        else {
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
        let (target_position, path_following, navigation) = update_mob_navigation_state(
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
        let has_blink = sim_config
            .ability_params
            .as_ref()
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
) -> (
    Vec<QueuedExplosionEvent>,
    Vec<QueuedPlayerMovementModifier>,
    SimTimings,
) {
    let mut queued_explosions = Vec::new();
    let mut queued_player_modifiers = Vec::new();
    let mut timings = SimTimings::default();
    let mut collision_cache = HashMap::<ChunkKey, CollisionChunkCacheEntry>::new();
    while *next_sim_ms <= now_ms && timings.sim_steps < ENTITY_SIM_STEP_MAX_PER_BROADCAST {
        collision_cache.clear();
        state
            .entity_store
            .simulate(*next_sim_ms, &state.content_registry, wasm_manager);
        let (mut step_explosions, mut step_player_modifiers) = simulate_mobs(
            state,
            &mut collision_cache,
            wasm_manager,
            *next_sim_ms,
            &mut timings,
        );
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
    use super::navigation::{mob_nav_find_path, mob_nav_has_line_of_sight, mob_nav_neighbor_steps};
    use super::physics::{
        resolve_mob_collision, resolve_walking_collision_with_steps, stick_mob_to_ground,
    };
    use super::*;
    use crate::content_registry::ContentRegistry;
    use crate::shared::spatial::ChunkCoord;
    use std::sync::Arc;

    fn test_registry() -> Arc<ContentRegistry> {
        let (registry, _pending) = crate::plugin_loader::create_full_registry();
        Arc::new(registry)
    }

    fn test_server_state_with_world() -> ServerState {
        let world = ServerWorldOverlay::from_chunk_payloads(
            crate::shared::voxel::BaseWorldKind::Empty,
            Vec::<([i32; 4], crate::shared::chunk_payload::ResolvedChunkPayload)>::new(),
            0,
            false,
            HashSet::new(),
        );
        ServerState::new(
            world,
            1,
            false,
            false,
            Instant::now(),
            test_registry(),
            Vec::new(),
        )
    }

    fn set_solid_voxel(state: &mut ServerState, x: i32, y: i32, z: i32, w: i32) {
        let _ = state.apply_world_voxel_edit(
            [
                ChunkCoord::from_num(x),
                ChunkCoord::from_num(y),
                ChunkCoord::from_num(z),
                ChunkCoord::from_num(w),
            ],
            BlockData::simple(0, 1),
        );
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
        let path = mob_nav_find_path(
            &state,
            &mut cache,
            start,
            goal,
            0.2,
            MobLocomotionMode::Walking,
        )
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
        let stepped =
            resolve_walking_collision_with_steps(&state, &mut cache, old_pos, attempted_pos, scale);
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
        let stuck = stick_mob_to_ground(
            &state,
            &mut cache,
            start,
            0.3,
            MOB_WALK_GROUND_STICK_MAX_DROP,
        );
        let dropped = start[1] - stuck[1];
        assert!(dropped <= MOB_WALK_GROUND_STICK_MAX_DROP + 1e-4);
        assert!(dropped + 1e-4 >= MOB_WALK_GROUND_STICK_MAX_DROP - MOB_WALK_GROUND_STICK_STEP);
    }
}
