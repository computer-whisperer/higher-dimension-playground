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
    // Prow pulses between sensor glow (4) and glow accent (2)
    let pulse_mat: u8 = if pulse > 0.66 { 4 } else { 2 };

    let core_offset = [0.0, 0.06 + bob, 0.0, 0.0];
    let mut parts = Vec::with_capacity(8);

    // Core body — chitin top/sides, belly underneath
    parts.push(part(
        core_offset,
        [s * 0.54, s * 0.52, s * 0.74, s * 0.56],
        [0, 0, 0, 3, 0, 0, 0, 0],
    ));

    // Prow — sensor/glow pulse
    parts.push(part(
        add4(core_offset, [0.0, 0.08, 0.47, 0.0]),
        [s * 0.30, s * 0.26, s * 0.30, s * 0.26],
        [
            pulse_mat, 2, pulse_mat, 2, pulse_mat, 2, pulse_mat, 2,
        ],
    ));

    // Fins (W-axis pair) — chitin with glow edges
    for w_sign in [1.0f32, -1.0] {
        parts.push(part(
            add4(core_offset, [0.0, 0.18, 0.05, 0.34 * w_sign]),
            [s * 0.14, s * 0.30, s * 0.44, s * 0.12],
            [0, 2, 0, 0, 0, 2, 0, 0],
        ));
    }

    // Pods (4 legs) — joints with chitin caps
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
            [1, 0, 1, 3, 1, 0, 1, 3],
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
    // Pulsing charge effect — core glows brighter periodically
    let charge = powf_approx((sinf(anim_t * 0.8) * 0.5 + 0.5), 1.8);
    let face_mat: u8 = if charge > 0.70 { 5 } else { 3 }; // lava-veined vs eyes

    let bob = 0.03 * sinf(anim_t * 0.7);
    let body_offset = [0.0, 0.02 + bob, 0.0, 0.0];
    let mut parts = Vec::with_capacity(9);

    // Torso — squat, dense, wider than tall
    //   +X/-X: hide, +Y: hide (top), -Y: dark (underside)
    //   +Z/-Z: hide/belly, +W/-W: hide
    parts.push(part(
        body_offset,
        [s * 0.68, s * 0.62, s * 0.72, s * 0.74],
        [0, 0, 0, 1, 0, 2, 0, 0],
    ));

    // Head — wider, flatter, sits forward and on top of torso
    //   Front face (+Z) gets the glowing eye/charge material
    let head_y = 0.48 + 0.04 * sinf(anim_t * 0.6);
    parts.push(part(
        add4(body_offset, [0.0, head_y, 0.14, 0.0]),
        [s * 0.52, s * 0.34, s * 0.46, s * 0.50],
        [0, 0, 1, 0, face_mat, 0, 0, 0],
    ));

    // Core — small glowing box inside the torso, the "bomb"
    let core_pulse = 1.0 + 0.12 * sinf(anim_t * 1.6);
    parts.push(part(
        add4(body_offset, [0.0, 0.06, 0.0, 0.0]),
        [s * 0.22 * core_pulse, s * 0.20 * core_pulse, s * 0.22 * core_pulse, s * 0.22 * core_pulse],
        [4, 4, 4, 4, 4, 4, 4, 4],
    ));

    // Dorsal ridge — dark raised strip along the top/back
    parts.push(part(
        add4(body_offset, [0.0, 0.42, -0.22, 0.0]),
        [s * 0.18, s * 0.10, s * 0.34, s * 0.16],
        [1, 1, 1, 1, 1, 1, 1, 1],
    ));

    // Jaw — small protrusion under the head, slightly lighter
    parts.push(part(
        add4(body_offset, [0.0, 0.22, 0.36, 0.0]),
        [s * 0.36, s * 0.10, s * 0.18, s * 0.34],
        [2, 2, 1, 2, 2, 2, 2, 2],
    ));

    // Legs (4) — thick, splayed, low to the ground
    let leg_specs: [([f32; 4], f32); 4] = [
        ([0.42, -0.42, 0.34, 0.30], 0.0),
        ([-0.42, -0.42, -0.34, -0.30], core::f32::consts::PI),
        ([0.42, -0.42, -0.34, 0.30], core::f32::consts::FRAC_PI_2),
        (
            [-0.42, -0.42, 0.34, -0.30],
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
            [s * 0.22, s * 0.30, s * 0.20, s * 0.20],
            [1, 1, 0, 1, 0, 0, 0, 0],
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

    // Body — carapace with phase energy accents and eye spots
    parts.push(part(
        body_offset,
        [s * 0.88, s * 0.36, s * 0.78, s * 0.96],
        [0, 0, 0, 2, 3, 2, 0, 0],
    ));

    // Core — dimensional core glow
    parts.push(part(
        add4(body_offset, [0.0, 0.11, 0.0, 0.0]),
        [s * 0.32, s * 0.30, s * 0.32, s * 0.32],
        [7, 7, 7, 7, 7, 7, 7, 7],
    ));

    // Legs (8) — webbing with carapace joints
    let splay = 0.12 * sinf(anim_t);
    let leg_y = -0.16 + 0.04 * cosf(anim_t * 1.5);
    let leg_specs: [([f32; 4], [f32; 4]); 8] = [
        (
            [0.56 + splay, leg_y, 0.42, 0.20],
            [0.72, 0.08, 0.14, 0.18],
        ),
        (
            [0.56 + splay, leg_y, -0.42, -0.20],
            [0.72, 0.08, 0.14, 0.18],
        ),
        (
            [-0.56 - splay, leg_y, 0.42, -0.20],
            [0.72, 0.08, 0.14, 0.18],
        ),
        (
            [-0.56 - splay, leg_y, -0.42, 0.20],
            [0.72, 0.08, 0.14, 0.18],
        ),
        (
            [0.20, leg_y, 0.52 + splay, 0.56],
            [0.16, 0.08, 0.72, 0.18],
        ),
        (
            [-0.20, leg_y, -0.52 - splay, -0.56],
            [0.16, 0.08, 0.72, 0.18],
        ),
        (
            [0.20, leg_y, -0.52 - splay, 0.56],
            [0.16, 0.08, 0.72, 0.18],
        ),
        (
            [-0.20, leg_y, 0.52 + splay, -0.56],
            [0.16, 0.08, 0.72, 0.18],
        ),
    ];
    for (offset, axis_scale) in leg_specs {
        let leg_offset = add4(body_offset, offset);
        parts.push(part(
            leg_offset,
            [
                s * axis_scale[0],
                s * axis_scale[1],
                s * axis_scale[2],
                s * axis_scale[3],
            ],
            [0, 1, 0, 1, 0, 1, 0, 1],
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
