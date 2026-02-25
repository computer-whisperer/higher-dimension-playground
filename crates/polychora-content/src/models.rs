use alloc::vec::Vec;
use libm::{cosf, sinf};
use polychora_plugin_api::content_ids;
use polychora_plugin_api::model_abi::{EntityModelInput, EntityModelOutput, EntityModelPart};

/// Dispatch entity model by (entity_ns, entity_type).
pub fn entity_model(input: &EntityModelInput) -> EntityModelOutput {
    match (input.entity_ns, input.entity_type) {
        (content_ids::CONTENT_NS, content_ids::ENTITY_CUBE) => cube_model(input),
        (content_ids::CONTENT_NS, content_ids::ENTITY_ROTOR) => rotor_model(input),
        (content_ids::CONTENT_NS, content_ids::ENTITY_DRIFTER) => drifter_model(input),
        (content_ids::CONTENT_NS, content_ids::ENTITY_SEEKER) => seeker_model(input),
        (content_ids::CONTENT_NS, content_ids::ENTITY_CREEPER) => creeper_model(input),
        (content_ids::CONTENT_NS, content_ids::ENTITY_PHASE_SPIDER) => phase_spider_model(input),
        _ => fallback_cube(input),
    }
}

fn part(offset: [f32; 4], half_extents: [f32; 4], cell_materials: [u8; 8]) -> EntityModelPart {
    EntityModelPart {
        offset,
        half_extents,
        cell_materials,
    }
}

fn fallback_cube(input: &EntityModelInput) -> EntityModelOutput {
    let s = input.scale;
    EntityModelOutput {
        parts: alloc::vec![part([0.0; 4], [s; 4], [0; 8])],
    }
}

fn cube_model(input: &EntityModelInput) -> EntityModelOutput {
    let s = input.scale;
    EntityModelOutput {
        parts: alloc::vec![part([0.0; 4], [s; 4], [0; 8])],
    }
}

fn rotor_model(input: &EntityModelInput) -> EntityModelOutput {
    let s = input.scale;
    EntityModelOutput {
        parts: alloc::vec![part(
            [0.0; 4],
            [s * 0.56, s * 0.56, s * 1.35, s * 0.82],
            [0, 0, 0, 1, 0, 1, 0, 1],
        )],
    }
}

fn drifter_model(input: &EntityModelInput) -> EntityModelOutput {
    let s = input.scale;
    EntityModelOutput {
        parts: alloc::vec![part(
            [0.0; 4],
            [s * 1.15, s * 0.44, s * 0.72, s * 1.05],
            [2, 0, 0, 0, 0, 2, 0, 0],
        )],
    }
}

fn seeker_model(input: &EntityModelInput) -> EntityModelOutput {
    let s = input.scale;
    let speed = input.speed_xzw;
    let stride = clamp(speed / 3.6, 0.0, 1.35);
    let anim_t =
        input.elapsed_s * (2.9 + stride * 4.8) + input.entity_id as f32 * 0.31;
    let bob = (0.02 + 0.035 * stride) * sinf(anim_t * 0.9);
    let pulse = powf_approx((sinf(anim_t * 1.1) * 0.5 + 0.5), 1.6);
    let pulse_bias: u8 = if pulse > 0.66 { 8 } else { 6 };

    let core_offset = [0.0, 0.06 + bob, 0.0, 0.0];
    let mut parts = Vec::with_capacity(8);

    // Core body
    parts.push(part(
        core_offset,
        [s * 0.54, s * 0.52, s * 0.74, s * 0.56],
        [0, 1, 2, 0, 3, 0, 4, 0],
    ));

    // Prow
    parts.push(part(
        add4(core_offset, [0.0, 0.08, 0.47, 0.0]),
        [s * 0.30, s * 0.26, s * 0.30, s * 0.26],
        [
            2, pulse_bias, 2, pulse_bias, 2, pulse_bias, 2, pulse_bias,
        ],
    ));

    // Fins (W-axis pair)
    for w_sign in [1.0f32, -1.0] {
        parts.push(part(
            add4(core_offset, [0.0, 0.18, 0.05, 0.34 * w_sign]),
            [s * 0.14, s * 0.30, s * 0.44, s * 0.12],
            [0, 5, 0, 2, 0, 5, 0, 2],
        ));
    }

    // Pods (4 legs)
    let pod_specs: [([f32; 4], f32); 4] = [
        ([0.34, -0.19, 0.18, 0.18], 0.0),
        ([-0.34, -0.19, -0.18, -0.18], core::f32::consts::PI),
        ([0.34, -0.19, -0.18, 0.18], core::f32::consts::FRAC_PI_2),
        (
            [-0.34, -0.19, 0.18, -0.18],
            core::f32::consts::PI + core::f32::consts::FRAC_PI_2,
        ),
    ];
    for (base_offset, phase) in pod_specs {
        let swing = sinf(anim_t * 1.8 + phase);
        let lift = max_f32(swing, 0.0);
        let offset = add4(core_offset, [
            base_offset[0],
            base_offset[1] + 0.08 * stride * lift,
            base_offset[2] + 0.11 * stride * swing,
            base_offset[3],
        ]);
        parts.push(part(
            offset,
            [s * 0.18, s * 0.24, s * 0.18, s * 0.16],
            [1, 0, 3, 0, 1, 0, 3, 0],
        ));
    }

    EntityModelOutput { parts }
}

fn creeper_model(input: &EntityModelInput) -> EntityModelOutput {
    let s = input.scale;
    let speed = input.speed_xzw;
    let stride = clamp(speed / 3.2, 0.0, 1.45);
    let anim_t =
        input.elapsed_s * (2.4 + stride * 4.2) + input.entity_id as f32 * 0.29;
    let charge = powf_approx((sinf(anim_t * 0.8) * 0.5 + 0.5), 1.8);
    let charge_bias: u8 = if charge > 0.70 { 9 } else { 7 };

    let body_offset = [0.0, 0.05 + 0.03 * sinf(anim_t * 0.7), 0.0, 0.0];
    let mut parts = Vec::with_capacity(7);

    // Body
    parts.push(part(
        body_offset,
        [s * 0.86, s * 0.94, s * 0.84, s * 0.98],
        [3, 1, 4, 0, 5, 2, 4, 0],
    ));

    // Head
    parts.push(part(
        add4(body_offset, [0.0, 0.58 + 0.04 * sinf(anim_t * 0.6), 0.0, 0.0]),
        [s * 0.52, s * 0.40, s * 0.50, s * 0.58],
        [
            2,
            charge_bias,
            2,
            charge_bias,
            2,
            charge_bias,
            2,
            charge_bias,
        ],
    ));

    // Vent
    parts.push(part(
        add4(body_offset, [0.0, 0.16, -0.40, 0.0]),
        [s * 0.46, s * 0.22, s * 0.24, s * 0.40],
        [6, 3, 6, 3, 6, 3, 6, 3],
    ));

    // Legs (4)
    let leg_specs: [([f32; 4], f32); 4] = [
        ([0.38, -0.56, 0.30, 0.30], 0.0),
        ([-0.38, -0.56, -0.30, -0.30], core::f32::consts::PI),
        ([0.38, -0.56, -0.30, 0.30], core::f32::consts::FRAC_PI_2),
        (
            [-0.38, -0.56, 0.30, -0.30],
            core::f32::consts::PI + core::f32::consts::FRAC_PI_2,
        ),
    ];
    for (base_offset, phase) in leg_specs {
        let swing = sinf(anim_t * 1.65 + phase);
        let lift = max_f32(swing, 0.0);
        let offset = add4(body_offset, [
            base_offset[0],
            base_offset[1] + 0.10 * stride * lift,
            base_offset[2] + 0.12 * stride * swing,
            base_offset[3],
        ]);
        parts.push(part(
            offset,
            [s * 0.26, s * 0.36, s * 0.24, s * 0.24],
            [0, 1, 0, 2, 0, 1, 0, 2],
        ));
    }

    EntityModelOutput { parts }
}

fn phase_spider_model(input: &EntityModelInput) -> EntityModelOutput {
    let s = input.scale;
    let anim_t = input.elapsed_s * 6.0 + input.entity_id as f32 * 0.23;
    let body_bob = 0.05 * sinf(anim_t * 0.7);

    let body_offset = [0.0, 0.08 + body_bob, 0.0, 0.0];
    let mut parts = Vec::with_capacity(10);

    // Body
    parts.push(part(
        body_offset,
        [s * 0.88, s * 0.36, s * 0.78, s * 0.96],
        [2, 0, 4, 0, 3, 0, 5, 0],
    ));

    // Core
    parts.push(part(
        add4(body_offset, [0.0, 0.11, 0.0, 0.0]),
        [s * 0.32, s * 0.30, s * 0.32, s * 0.32],
        [7, 9, 8, 9, 7, 9, 8, 9],
    ));

    // Legs (8)
    let splay = 0.12 * sinf(anim_t);
    let leg_y = -0.16 + 0.04 * cosf(anim_t * 1.5);
    let leg_specs: [([f32; 4], [f32; 4], u8); 8] = [
        (
            [0.56 + splay, leg_y, 0.42, 0.20],
            [0.72, 0.08, 0.14, 0.18],
            1,
        ),
        (
            [0.56 + splay, leg_y, -0.42, -0.20],
            [0.72, 0.08, 0.14, 0.18],
            2,
        ),
        (
            [-0.56 - splay, leg_y, 0.42, -0.20],
            [0.72, 0.08, 0.14, 0.18],
            3,
        ),
        (
            [-0.56 - splay, leg_y, -0.42, 0.20],
            [0.72, 0.08, 0.14, 0.18],
            4,
        ),
        (
            [0.20, leg_y, 0.52 + splay, 0.56],
            [0.16, 0.08, 0.72, 0.18],
            5,
        ),
        (
            [-0.20, leg_y, -0.52 - splay, -0.56],
            [0.16, 0.08, 0.72, 0.18],
            6,
        ),
        (
            [0.20, leg_y, -0.52 - splay, 0.56],
            [0.16, 0.08, 0.72, 0.18],
            7,
        ),
        (
            [-0.20, leg_y, 0.52 + splay, -0.56],
            [0.16, 0.08, 0.72, 0.18],
            8,
        ),
    ];
    for (offset, axis_scale, material_bias) in leg_specs {
        let leg_offset = add4(body_offset, offset);
        parts.push(part(
            leg_offset,
            [
                s * axis_scale[0],
                s * axis_scale[1],
                s * axis_scale[2],
                s * axis_scale[3],
            ],
            [
                0,
                material_bias,
                0,
                material_bias,
                0,
                material_bias,
                0,
                material_bias,
            ],
        ));
    }

    EntityModelOutput { parts }
}

// --- Utility functions ---

fn add4(a: [f32; 4], b: [f32; 4]) -> [f32; 4] {
    [a[0] + b[0], a[1] + b[1], a[2] + b[2], a[3] + b[3]]
}

fn clamp(x: f32, min: f32, max: f32) -> f32 {
    if x < min {
        min
    } else if x > max {
        max
    } else {
        x
    }
}

fn max_f32(a: f32, b: f32) -> f32 {
    if a > b { a } else { b }
}

/// Approximate x^p for small positive x and reasonable p.
fn powf_approx(x: f32, p: f32) -> f32 {
    libm::expf(p * libm::logf(if x > 0.0 { x } else { 1e-10 }))
}
