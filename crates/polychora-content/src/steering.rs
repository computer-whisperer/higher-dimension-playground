use crate::math4d::{distance4_sq, normalize4_or_default};
use libm::{cosf, sinf, sqrtf};
use polychora_plugin_api::content_ids;
use polychora_plugin_api::mob_abi::{
    MobAbilityCheck, MobAbilityResult, MobSteeringInput, MobSteeringOutput,
};

/// Path-following reach distance (must match MOB_NAV_PATH_NODE_REACH_DISTANCE
/// in the host).
const PATH_NODE_REACH_DISTANCE: f32 = 0.62;

/// Dispatch steering by entity type.
pub fn mob_steering(input: &MobSteeringInput) -> MobSteeringOutput {
    match (input.entity_ns, input.entity_type) {
        (content_ids::CONTENT_NS, content_ids::ENTITY_SEEKER) => seeker_step(input),
        (content_ids::CONTENT_NS, content_ids::ENTITY_CREEPER) => creeper_step(input),
        (content_ids::CONTENT_NS, content_ids::ENTITY_PHASE_SPIDER) => phase_spider_step(input),
        _ => seeker_step(input),
    }
}

/// Evaluate an ability trigger for a mob.
pub fn mob_ability_check(check: &MobAbilityCheck) -> MobAbilityResult {
    let should_trigger = match check {
        MobAbilityCheck::Detonate {
            nearest_player_distance,
            trigger_distance,
            ..
        } => *trigger_distance > 0.0 && *nearest_player_distance <= *trigger_distance,
        MobAbilityCheck::Blink {
            has_target,
            path_following,
            now_ms,
            next_phase_ms,
            blocked_progress_epsilon,
            attempted_move_distance,
            resolved_move_distance,
            ..
        } => {
            let blocked = *attempted_move_distance > 0.12
                && *resolved_move_distance + *blocked_progress_epsilon < *attempted_move_distance;
            *now_ms >= *next_phase_ms && *has_target && (*path_following || blocked)
        }
    };
    MobAbilityResult { should_trigger }
}

fn seeker_step(input: &MobSteeringInput) -> MobSteeringOutput {
    let t_s = input.now_ms as f32 * 0.001;
    let (desired_dir, slow_factor) = if let Some(target) = input.target_position {
        let to_target = [
            target[0] - input.position[0],
            target[1] - input.position[1],
            target[2] - input.position[2],
            target[3] - input.position[3],
        ];
        let distance = sqrtf(distance4_sq(target, input.position));
        let direct = normalize4_or_default(to_target, [0.0, 0.0, 1.0, 0.0]);
        if input.path_following {
            let slow = if distance < PATH_NODE_REACH_DISTANCE * 0.9 {
                0.62
            } else {
                1.05
            };
            (direct, slow)
        } else if input.simple_steer {
            let slow = if distance < input.preferred_distance * 0.6 {
                0.42
            } else {
                1.0
            };
            (direct, slow)
        } else {
            let tangent = normalize4_or_default(
                [-direct[2], -direct[3], direct[0], direct[1]],
                [direct[3], 0.0, -direct[0], direct[1]],
            );
            let weave_phase = t_s * 1.7 + input.phase_offset;
            let weave = [
                0.10 * sinf(weave_phase),
                0.08 * cosf(weave_phase * 0.7),
                -0.10 * sinf(weave_phase),
                0.08 * sinf(weave_phase * 1.3),
            ];
            let pursuit = if distance > input.preferred_distance {
                1.0
            } else {
                -0.45
            };
            let slow = if distance < input.preferred_distance * 0.6 {
                0.42
            } else {
                1.0
            };
            let desired = normalize4_or_default(
                [
                    direct[0] * pursuit + tangent[0] * input.tangent_weight + weave[0],
                    direct[1] * pursuit + tangent[1] * input.tangent_weight + weave[1],
                    direct[2] * pursuit + tangent[2] * input.tangent_weight + weave[2],
                    direct[3] * pursuit + tangent[3] * input.tangent_weight + weave[3],
                ],
                direct,
            );
            (desired, slow)
        }
    } else {
        let phase = t_s * 0.65 + input.phase_offset;
        (
            normalize4_or_default(
                [
                    cosf(phase),
                    0.45 * sinf(phase * 0.7),
                    sinf(phase),
                    cosf(phase * 1.1),
                ],
                [0.0, 0.0, 1.0, 0.0],
            ),
            0.35,
        )
    };

    MobSteeringOutput {
        desired_direction: desired_dir,
        speed_factor: slow_factor,
    }
}

fn creeper_step(input: &MobSteeringInput) -> MobSteeringOutput {
    let t_s = input.now_ms as f32 * 0.001;
    let (desired_dir, speed_factor) = if let Some(target) = input.target_position {
        let to_target = [
            target[0] - input.position[0],
            target[1] - input.position[1],
            target[2] - input.position[2],
            target[3] - input.position[3],
        ];
        let distance = sqrtf(distance4_sq(target, input.position));
        let direct = normalize4_or_default(to_target, [0.0, 0.0, 1.0, 0.0]);
        if input.path_following {
            let speed = if distance > PATH_NODE_REACH_DISTANCE * 1.2 {
                1.18
            } else {
                0.66
            };
            (direct, speed)
        } else if input.simple_steer {
            let speed = if distance > input.preferred_distance * 0.9 {
                1.15
            } else {
                0.70
            };
            (direct, speed)
        } else {
            let orbit_a = normalize4_or_default(
                [-direct[2], direct[3], direct[0], -direct[1]],
                [direct[3], 0.0, -direct[0], direct[1]],
            );
            let orbit_b = normalize4_or_default(
                [direct[1], -direct[0], direct[3], -direct[2]],
                [0.0, direct[2], -direct[1], direct[0]],
            );
            let phase = t_s * 2.35 + input.phase_offset;
            let orbit_mix = [
                orbit_a[0] * sinf(phase) + orbit_b[0] * cosf(phase),
                orbit_a[1] * sinf(phase) + orbit_b[1] * cosf(phase),
                orbit_a[2] * sinf(phase) + orbit_b[2] * cosf(phase),
                orbit_a[3] * sinf(phase) + orbit_b[3] * cosf(phase),
            ];
            let too_close = distance < input.preferred_distance * 0.55;
            let in_lunge_band = distance > input.preferred_distance * 0.80
                && distance < input.preferred_distance * 1.95;
            let lunge = in_lunge_band && sinf(phase) > 0.58;
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
                    direct[0] * pressure + orbit_mix[0] * input.tangent_weight,
                    direct[1] * pressure + orbit_mix[1] * input.tangent_weight,
                    direct[2] * pressure + orbit_mix[2] * input.tangent_weight,
                    direct[3] * pressure + orbit_mix[3] * input.tangent_weight,
                ],
                direct,
            );
            (desired, speed)
        }
    } else {
        let phase = t_s * 0.72 + input.phase_offset;
        (
            normalize4_or_default(
                [
                    0.7 * sinf(phase),
                    cosf(phase * 0.9),
                    0.7 * cosf(phase),
                    sinf(phase * 1.4),
                ],
                [0.0, 0.0, 1.0, 0.0],
            ),
            0.38,
        )
    };

    MobSteeringOutput {
        desired_direction: desired_dir,
        speed_factor,
    }
}

fn phase_spider_step(input: &MobSteeringInput) -> MobSteeringOutput {
    let t_s = input.now_ms as f32 * 0.001;
    let (desired_dir, speed_factor) = if let Some(target) = input.target_position {
        let to_target = [
            target[0] - input.position[0],
            target[1] - input.position[1],
            target[2] - input.position[2],
            target[3] - input.position[3],
        ];
        let distance = sqrtf(distance4_sq(target, input.position));
        let direct = normalize4_or_default(to_target, [0.0, 0.0, 1.0, 0.0]);
        if input.path_following {
            let phase = t_s * 6.8 + input.phase_offset;
            let tangent = normalize4_or_default(
                [direct[3], 0.0, -direct[0], direct[2]],
                [-direct[2], direct[3], direct[0], -direct[1]],
            );
            let lateral_scale = if input.simple_steer { 0.0 } else { 0.18 };
            let desired = normalize4_or_default(
                [
                    direct[0] + tangent[0] * lateral_scale * sinf(phase),
                    direct[1] + tangent[1] * lateral_scale * sinf(phase * 0.7),
                    direct[2] + tangent[2] * lateral_scale * sinf(phase),
                    direct[3] + tangent[3] * lateral_scale * cosf(phase * 0.9),
                ],
                direct,
            );
            let speed = if distance > PATH_NODE_REACH_DISTANCE * 1.4 {
                1.32
            } else {
                0.78
            };
            (desired, speed)
        } else if input.simple_steer {
            let speed = if distance > input.preferred_distance {
                1.20
            } else {
                0.74
            };
            (direct, speed)
        } else {
            let phase = t_s * 5.2 + input.phase_offset;
            let strafe_a = normalize4_or_default(
                [-direct[2], direct[3], direct[0], -direct[1]],
                [direct[3], 0.0, -direct[0], direct[1]],
            );
            let strafe_b = normalize4_or_default(
                [direct[1], -direct[0], direct[3], -direct[2]],
                [0.0, direct[2], -direct[1], direct[0]],
            );
            let strafe_mix = [
                strafe_a[0] * sinf(phase) + strafe_b[0] * cosf(phase),
                strafe_a[1] * sinf(phase) + strafe_b[1] * cosf(phase),
                strafe_a[2] * sinf(phase) + strafe_b[2] * cosf(phase),
                strafe_a[3] * sinf(phase) + strafe_b[3] * cosf(phase),
            ];
            let stalk = if distance > input.preferred_distance * 1.35 {
                1.0
            } else if distance < input.preferred_distance * 0.65 {
                -0.52
            } else {
                0.24
            };
            let vertical = 0.11 * sinf(phase * 1.4);
            let desired = normalize4_or_default(
                [
                    direct[0] * stalk + strafe_mix[0] * input.tangent_weight * 1.30,
                    direct[1] * stalk + strafe_mix[1] * input.tangent_weight * 1.30 + vertical,
                    direct[2] * stalk + strafe_mix[2] * input.tangent_weight * 1.30,
                    direct[3] * stalk + strafe_mix[3] * input.tangent_weight * 1.30,
                ],
                direct,
            );
            let speed = if distance > input.preferred_distance {
                1.15
            } else {
                0.70
            };
            (desired, speed)
        }
    } else {
        let phase = t_s * 0.93 + input.phase_offset;
        (
            normalize4_or_default(
                [
                    0.9 * sinf(phase),
                    0.28 * cosf(phase * 1.1),
                    0.9 * cosf(phase),
                    sinf(phase * 1.6),
                ],
                [0.0, 0.0, 1.0, 0.0],
            ),
            0.44,
        )
    };

    MobSteeringOutput {
        desired_direction: desired_dir,
        speed_factor,
    }
}
