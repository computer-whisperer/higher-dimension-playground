use super::*;

fn dot4(a: [f32; 4], b: [f32; 4]) -> f32 {
    a[0] * b[0] + a[1] * b[1] + a[2] * b[2] + a[3] * b[3]
}

fn len4(a: [f32; 4]) -> f32 {
    dot4(a, a).sqrt()
}

fn nearest_3d_pose_error(cam: &Camera4D) -> f32 {
    let mut best = f32::INFINITY;
    for hidden_axis in 0..4 {
        let mut error = cam.look_forward[hidden_axis].abs()
            + cam.look_up[hidden_axis].abs()
            + cam.look_right[hidden_axis].abs()
            + (1.0 - cam.look_side[hidden_axis].abs());
        for axis in 0..4 {
            if axis != hidden_axis {
                error += cam.look_side[axis].abs();
            }
        }
        if error < best {
            best = error;
        }
    }
    best
}

#[test]
fn look_direction_matches_view_rotation_inverse() {
    let mut cam = Camera4D::new();
    cam.yaw = 0.37;
    cam.pitch = -0.22;
    cam.xw_angle = 0.61;
    cam.zw_angle = -0.48;
    cam.yw_deviation = 0.19;

    let got = cam.look_direction();
    let view = cam.view_matrix();
    let look_view = [
        0.0,
        0.0,
        std::f32::consts::FRAC_1_SQRT_2,
        std::f32::consts::FRAC_1_SQRT_2,
    ];
    let mut expected = [0.0f32; 4];

    // For pure directions, inverse(view_rotation) = transpose(view_rotation).
    for world_axis in 0..4 {
        let mut value = 0.0f32;
        for view_axis in 0..4 {
            value += view[[view_axis, world_axis]] * look_view[view_axis];
        }
        expected[world_axis] = value;
    }

    let len = (expected[0] * expected[0]
        + expected[1] * expected[1]
        + expected[2] * expected[2]
        + expected[3] * expected[3])
        .sqrt();
    for v in &mut expected {
        *v /= len;
    }

    for axis in 0..4 {
        assert!((got[axis] - expected[axis]).abs() < 1e-4);
    }
}

#[test]
fn mouse_right_turns_toward_positive_x_in_standard_mode() {
    let mut cam = Camera4D::new();
    cam.yaw = 0.0;
    cam.pitch = 0.0;
    cam.xw_angle = 0.0;
    cam.zw_angle = 0.0;
    cam.yw_deviation = 0.0;

    let before_x = cam.look_direction()[0];
    cam.apply_mouse_look_on(40.0, 0.0, 0.01, AngleTarget::Yaw, AngleTarget::Pitch);
    let after_x = cam.look_direction()[0];
    assert!(
            after_x > before_x,
            "mouse-right should increase look.x in standard yaw mode: before={before_x:.4} after={after_x:.4}"
        );
}

#[test]
fn mouse_up_looks_up_in_baseline_orientation() {
    let mut cam = Camera4D::new();
    cam.yaw = 0.0;
    cam.pitch = 0.0;
    cam.xw_angle = 0.0;
    cam.zw_angle = 0.0;
    cam.yw_deviation = 0.0;

    let before_y = cam.look_direction()[1];
    cam.apply_mouse_look_on(0.0, -40.0, 0.01, AngleTarget::Yaw, AngleTarget::Pitch);
    let after_y = cam.look_direction()[1];
    assert!(
            after_y > before_y,
            "mouse-up should increase look.y when not Y-inverted: before={before_y:.4} after={after_y:.4}"
        );
}

#[test]
fn movement_forward_stays_on_xzw_plane() {
    let mut cam = Camera4D::new();
    cam.position = [0.0, 0.0, 0.0, 0.0];
    cam.yaw = 0.37;
    cam.pitch = -0.22;
    cam.xw_angle = 0.61;
    cam.zw_angle = -0.48;
    cam.yw_deviation = 0.19;
    cam.is_flying = true;

    cam.apply_movement(1.0, 0.0, 0.0, 0.0, 1.0, 1.0);
    assert!(
        cam.position[1].abs() < 1e-6,
        "forward movement should not change Y; got {}",
        cam.position[1]
    );
    let xzw_len = (cam.position[0] * cam.position[0]
        + cam.position[2] * cam.position[2]
        + cam.position[3] * cam.position[3])
        .sqrt();
    assert!(
        (xzw_len - 1.0).abs() < 1e-4,
        "forward movement should be unit speed on XZW plane, len={xzw_len}"
    );
}

#[test]
fn upright_constraints_remove_roll_twist_and_keep_y_non_inverted() {
    let mut cam = Camera4D::new();
    cam.pitch = 2.0;
    cam.zw_angle = 0.73;
    cam.yw_deviation = -1.11;
    cam.enforce_upright_constraints();

    assert!((cam.zw_angle - 0.73).abs() < 1e-6);
    assert_eq!(cam.yw_deviation, 0.0);
    let max_pitch = 81.0_f32.to_radians();
    assert!(cam.pitch <= max_pitch + 1e-6);
    assert!(cam.pitch >= -max_pitch - 1e-6);
    assert!(
        !cam.is_y_inverted_upright(),
        "upright constraints should keep camera from Y inversion"
    );
}

#[test]
fn upright_forward_movement_matches_center_look_direction() {
    let mut cam = Camera4D::new();
    cam.position = [0.0, 0.0, 0.0, 0.0];
    cam.yaw = 0.41;
    cam.pitch = -0.33;
    cam.xw_angle = 0.77;
    cam.zw_angle = 0.92;
    cam.yw_deviation = -1.13;
    cam.is_flying = true;
    cam.enforce_upright_constraints();

    let expected = Camera4D::normalize_xzw(cam.look_direction_upright());
    cam.apply_movement_upright(1.0, 0.0, 0.0, 0.0, 1.0, 1.0);
    for axis in 0..4 {
        assert!(
            (cam.position[axis] - expected[axis]).abs() < 1e-4,
            "axis={axis} got={} expected={}",
            cam.position[axis],
            expected[axis]
        );
    }
}

#[test]
fn flying_vertical_input_moves_only_world_y() {
    let mut cam = Camera4D::new();
    cam.position = [0.0, 0.0, 0.0, 0.0];
    cam.yaw = 0.52;
    cam.pitch = -0.63;
    cam.xw_angle = 1.11;
    cam.zw_angle = -0.44;
    cam.is_flying = true;
    cam.enforce_upright_constraints();

    cam.apply_movement_upright(0.0, 0.0, 1.0, 0.0, 1.0, 1.0);

    assert!(
        cam.position[1] > 0.9999,
        "expected positive world-Y movement, got y={}",
        cam.position[1]
    );
    assert!(
        cam.position[0].abs() < 1e-4,
        "expected x≈0, got x={}",
        cam.position[0]
    );
    assert!(
        cam.position[2].abs() < 1e-4,
        "expected z≈0, got z={}",
        cam.position[2]
    );
    assert!(
        cam.position[3].abs() < 1e-4,
        "expected w≈0, got w={}",
        cam.position[3]
    );
}

#[test]
fn upright_movement_normalizes_diagonals() {
    let mut cam = Camera4D::new();
    cam.position = [0.0, 0.0, 0.0, 0.0];
    cam.yaw = 0.2;
    cam.pitch = -0.1;
    cam.xw_angle = 0.3;
    cam.enforce_upright_constraints();
    cam.is_flying = true;

    cam.apply_movement_upright(1.0, 1.0, 1.0, 1.0, 1.0, 1.0);
    let len = (cam.position[0] * cam.position[0]
        + cam.position[1] * cam.position[1]
        + cam.position[2] * cam.position[2]
        + cam.position[3] * cam.position[3])
        .sqrt();
    assert!(
        (len - 1.0).abs() < 1e-4,
        "diagonal input should still move at unit speed: len={len}"
    );
}

#[test]
fn upright_yaw_does_not_invert_across_xw_sweep() {
    let mut cam = Camera4D::new();
    cam.pitch = 0.0;
    cam.enforce_upright_constraints();

    for step in -24..=24 {
        cam.xw_angle = step as f32 * (std::f32::consts::PI / 24.0);
        cam.yaw = 0.0;

        let rotation = cam.rotation_matrix_upright();
        let right = Camera4D::normalize_dir(Camera4D::world_direction_from_view_with_rotation(
            &rotation,
            [1.0, 0.0, 0.0, 0.0],
        ));

        let before = cam.look_direction_upright();
        cam.adjust_angle(AngleTarget::Yaw, -1e-4); // mouse-right equivalent delta
        let after = cam.look_direction_upright();

        let delta = [
            after[0] - before[0],
            after[1] - before[1],
            after[2] - before[2],
            after[3] - before[3],
        ];
        let along_right =
            delta[0] * right[0] + delta[1] * right[1] + delta[2] * right[2] + delta[3] * right[3];
        assert!(
            along_right > 0.0,
            "upright yaw should not invert at xw={:.2} deg (delta·right={:.6e})",
            cam.xw_angle.to_degrees(),
            along_right
        );
    }
}

#[test]
fn upright_zw_can_aim_center_ray_to_positive_z_axis() {
    let mut cam = Camera4D::new();
    cam.yaw = 0.0;
    cam.pitch = 0.0;
    cam.xw_angle = 0.0;
    cam.zw_angle = -std::f32::consts::FRAC_PI_2;
    cam.enforce_upright_constraints();

    let look = cam.look_direction_upright();
    assert!(look[2] > 0.9999, "expected +Z aim, got z={}", look[2]);
    assert!(look[0].abs() < 1e-4, "expected x≈0, got x={}", look[0]);
    assert!(look[1].abs() < 1e-4, "expected y≈0, got y={}", look[1]);
    assert!(look[3].abs() < 1e-4, "expected w≈0, got w={}", look[3]);
}

#[test]
fn upright_zero_angles_look_along_positive_x_axis() {
    let mut cam = Camera4D::new();
    cam.yaw = 0.0;
    cam.pitch = 0.0;
    cam.xw_angle = 0.0;
    cam.zw_angle = 0.0;
    cam.enforce_upright_constraints();

    let look = cam.look_direction_upright();
    assert!(look[0] > 0.9999, "expected +X aim, got x={}", look[0]);
    assert!(look[1].abs() < 1e-4, "expected y≈0, got y={}", look[1]);
    assert!(look[2].abs() < 1e-4, "expected z≈0, got z={}", look[2]);
    assert!(look[3].abs() < 1e-4, "expected w≈0, got w={}", look[3]);
}

#[test]
fn periodic_angles_are_wrapped_including_yw_deviation() {
    let mut cam = Camera4D::new();
    let near_pi = std::f32::consts::PI - 1e-3;

    cam.yaw = near_pi;
    cam.adjust_angle(AngleTarget::Yaw, 0.01);
    assert!(
        cam.yaw < 0.0 && cam.yaw.abs() < std::f32::consts::PI,
        "yaw should wrap into (-pi, pi], got {}",
        cam.yaw
    );

    cam.xw_angle = near_pi;
    cam.adjust_angle(AngleTarget::XwAngle, 0.01);
    assert!(
        cam.xw_angle < 0.0 && cam.xw_angle.abs() < std::f32::consts::PI,
        "xw should wrap into (-pi, pi], got {}",
        cam.xw_angle
    );

    cam.zw_angle = near_pi;
    cam.adjust_angle(AngleTarget::ZwAngle, 0.01);
    assert!(
        cam.zw_angle < 0.0 && cam.zw_angle.abs() < std::f32::consts::PI,
        "zw should wrap into (-pi, pi], got {}",
        cam.zw_angle
    );

    cam.yw_deviation = near_pi;
    cam.adjust_angle(AngleTarget::YwDeviation, 0.01);
    assert!(
        cam.yw_deviation < 0.0 && cam.yw_deviation.abs() < std::f32::consts::PI,
        "yw deviation should wrap into (-pi, pi], got {}",
        cam.yw_deviation
    );
}

#[test]
fn view_basis_vectors_remain_orthonormal_in_standard_mode() {
    let mut cam = Camera4D::new();
    cam.yaw = 0.74;
    cam.pitch = -0.43;
    cam.xw_angle = 1.16;
    cam.zw_angle = -0.82;
    cam.yw_deviation = 0.57;

    let (right, up, view_z, view_w) = cam.view_basis();
    let basis = [right, up, view_z, view_w];

    for i in 0..4 {
        let norm = len4(basis[i]);
        assert!(
            (norm - 1.0).abs() < 1e-4,
            "basis vector {i} should be unit length, got {norm}"
        );
        for j in (i + 1)..4 {
            let d = dot4(basis[i], basis[j]);
            assert!(
                d.abs() < 1e-4,
                "basis vectors {i},{j} should be orthogonal, dot={d}"
            );
        }
    }
}

#[test]
fn upright_forward_and_hidden_side_axes_are_orthogonal_unit_vectors() {
    let mut cam = Camera4D::new();
    cam.yaw = -0.62;
    cam.pitch = 0.28;
    cam.xw_angle = 1.03;
    cam.zw_angle = -0.51;
    cam.enforce_upright_constraints();

    let (_, _, view_z, view_w) = cam.view_basis_upright();
    let forward = Camera4D::normalize_dir([
        view_z[0] + view_w[0],
        view_z[1] + view_w[1],
        view_z[2] + view_w[2],
        view_z[3] + view_w[3],
    ]);
    let hidden_side = Camera4D::normalize_dir([
        view_w[0] - view_z[0],
        view_w[1] - view_z[1],
        view_w[2] - view_z[2],
        view_w[3] - view_z[3],
    ]);
    let dot = dot4(forward, hidden_side);
    let forward_len = len4(forward);
    let side_len = len4(hidden_side);

    assert!(
        dot.abs() < 1e-4,
        "forward and hidden-side axes should be orthogonal, dot={dot}"
    );
    assert!(
        (forward_len - 1.0).abs() < 1e-4,
        "forward axis should be unit length, got {forward_len}"
    );
    assert!(
        (side_len - 1.0).abs() < 1e-4,
        "hidden-side axis should be unit length, got {side_len}"
    );
}

#[test]
fn gravity_upright_forward_speed_is_pitch_invariant_on_xzw_hyperplane() {
    let mut flat = Camera4D::new();
    flat.position = [0.0, 0.0, 0.0, 0.0];
    flat.yaw = 0.18;
    flat.pitch = 0.0;
    flat.xw_angle = 0.63;
    flat.zw_angle = -0.21;
    flat.enforce_upright_constraints();
    flat.is_flying = false;

    let mut tilted = Camera4D::new();
    tilted.position = [0.0, 0.0, 0.0, 0.0];
    tilted.yaw = flat.yaw;
    tilted.pitch = 0.78;
    tilted.xw_angle = flat.xw_angle;
    tilted.zw_angle = flat.zw_angle;
    tilted.enforce_upright_constraints();
    tilted.is_flying = false;

    flat.apply_movement_upright(1.0, 0.0, 0.0, 0.0, 1.0, 1.0);
    tilted.apply_movement_upright(1.0, 0.0, 0.0, 0.0, 1.0, 1.0);

    let flat_xzw = (flat.position[0] * flat.position[0]
        + flat.position[2] * flat.position[2]
        + flat.position[3] * flat.position[3])
        .sqrt();
    let tilted_xzw = (tilted.position[0] * tilted.position[0]
        + tilted.position[2] * tilted.position[2]
        + tilted.position[3] * tilted.position[3])
        .sqrt();

    assert!(
        (flat_xzw - tilted_xzw).abs() < 1e-4,
        "gravity-mode speed in XZW should not depend on pitch: flat={flat_xzw} tilted={tilted_xzw}"
    );
    assert!(
        flat.position[1].abs() < 1e-6,
        "gravity movement should not change Y, got {}",
        flat.position[1]
    );
    assert!(
        tilted.position[1].abs() < 1e-6,
        "gravity movement should not change Y, got {}",
        tilted.position[1]
    );
}

#[test]
fn look_transport_zero_orientation_looks_along_positive_x() {
    let cam = Camera4D::new();
    let look = cam.look_direction_look_frame();
    assert!(look[0] > 0.9999, "expected +X aim, got x={}", look[0]);
    assert!(look[1].abs() < 1e-4, "expected y≈0, got y={}", look[1]);
    assert!(look[2].abs() < 1e-4, "expected z≈0, got z={}", look[2]);
    assert!(look[3].abs() < 1e-4, "expected w≈0, got w={}", look[3]);
}

#[test]
fn look_transport_half_turn_yaw_faces_negative_x() {
    let mut cam = Camera4D::new();
    cam.apply_mouse_look_transport(std::f32::consts::PI, 0.0, 1.0, false);
    let look = cam.look_direction_look_frame();
    assert!(look[0] < -0.9999, "expected -X aim, got x={}", look[0]);
    assert!(look[1].abs() < 1e-4, "expected y≈0, got y={}", look[1]);
    assert!(look[2].abs() < 1e-4, "expected z≈0, got z={}", look[2]);
    assert!(look[3].abs() < 1e-4, "expected w≈0, got w={}", look[3]);
}

#[test]
fn look_transport_hidden_mode_reaches_positive_w() {
    let mut cam = Camera4D::new();
    cam.apply_mouse_look_transport(std::f32::consts::FRAC_PI_2, 0.0, 1.0, true);
    let look = cam.look_direction_look_frame();
    assert!(look[3] > 0.9999, "expected +W aim, got w={}", look[3]);
    assert!(look[0].abs() < 1e-4, "expected x≈0, got x={}", look[0]);
}

#[test]
fn look_transport_non_hidden_vertical_uses_visual_pitch() {
    let mut cam = Camera4D::new();
    cam.apply_mouse_look_transport(0.0, -std::f32::consts::FRAC_PI_2, 1.0, false);
    let look = cam.look_direction_look_frame();
    assert!(look[1] > 0.9999, "expected +Y aim, got y={}", look[1]);
    assert!(
        look[3].abs() < 1e-4,
        "expected w≈0 for pure pitch, got w={}",
        look[3]
    );
}

#[test]
fn look_transport_hidden_mode_vertical_rotates_hidden_side_axis() {
    let mut cam = Camera4D::new();
    cam.apply_mouse_look_transport(0.0, -std::f32::consts::FRAC_PI_2, 1.0, true);
    let look = cam.look_direction_look_frame();
    assert!(
        look[0] > 0.9999,
        "expected look to stay +X, got x={}",
        look[0]
    );
    assert!(look[1].abs() < 1e-4, "expected y≈0, got y={}", look[1]);
    assert!(look[2].abs() < 1e-4, "expected z≈0, got z={}", look[2]);
    assert!(look[3].abs() < 1e-4, "expected w≈0, got w={}", look[3]);

    let (_, _, view_z, view_w) = cam.view_basis_look_frame();
    let hidden_side = Camera4D::normalize_dir([
        view_w[0] - view_z[0],
        view_w[1] - view_z[1],
        view_w[2] - view_z[2],
        view_w[3] - view_z[3],
    ]);
    assert!(
        hidden_side[2] < -0.9999,
        "expected hidden side near -Z, got z={}",
        hidden_side[2]
    );
    assert!(
        hidden_side[0].abs() < 1e-4,
        "expected hidden side x≈0, got x={}",
        hidden_side[0]
    );
    assert!(
        hidden_side[1].abs() < 1e-4,
        "expected hidden side y≈0, got y={}",
        hidden_side[1]
    );
    assert!(
        hidden_side[3].abs() < 1e-4,
        "expected hidden side w≈0, got w={}",
        hidden_side[3]
    );
}

#[test]
fn look_transport_forward_mode_uses_remaining_rotation_planes() {
    let mut cam = Camera4D::new();
    cam.apply_mouse_look_transport_with_modifiers(
        std::f32::consts::FRAC_PI_2,
        -std::f32::consts::FRAC_PI_2,
        1.0,
        false,
        true,
    );

    let look = cam.look_direction_look_frame();
    assert!(
        look[0] > 0.9999,
        "forward-mode remaining-plane turns should keep look near +X, got x={}",
        look[0]
    );
    assert!(look[1].abs() < 1e-4, "expected y≈0, got y={}", look[1]);
    assert!(look[2].abs() < 1e-4, "expected z≈0, got z={}", look[2]);
    assert!(look[3].abs() < 1e-4, "expected w≈0, got w={}", look[3]);

    let up = cam.look_up;
    assert!(
        up[3].abs() > 0.999,
        "forward-mode vertical should rotate up toward side axis, got up.w={}",
        up[3]
    );
}

#[test]
fn look_transport_forward_mode_disables_upright_stabilization() {
    let mut cam = Camera4D::new();
    cam.apply_mouse_look_transport_with_modifiers(
        0.0,
        -std::f32::consts::FRAC_PI_2,
        1.0,
        false,
        true,
    );
    let up_before = cam.look_up;

    for _ in 0..60 {
        cam.apply_mouse_look_transport_with_modifiers(0.0, 0.0, 1.0, false, true);
    }
    let up_after = cam.look_up;
    let alignment = dot4(up_before, up_after);
    assert!(
        alignment > 0.999,
        "forward-mode should not be pulled by upright stabilization, alignment={alignment}"
    );
}

#[test]
fn look_transport_yaw_does_not_change_look_y_after_pitch() {
    let mut cam = Camera4D::new();
    cam.apply_mouse_look_transport(0.0, -120.0, 0.002, false);
    let before = cam.look_direction_look_frame();
    cam.apply_mouse_look_transport(180.0, 0.0, 0.002, false);
    let after = cam.look_direction_look_frame();
    assert!(
        (after[1] - before[1]).abs() < 1e-4,
        "yaw should preserve look.y in LOOK-TR: before={} after={}",
        before[1],
        after[1]
    );
}

#[test]
fn look_transport_roll_damping_reduces_tilt() {
    let mut cam = Camera4D::new();
    Camera4D::rotate_axis_pair(
        &mut cam.look_up,
        &mut cam.look_right,
        std::f32::consts::FRAC_PI_2 * 0.8,
    );
    cam.reorthonormalize_look_frame();

    let world_up = [0.0, 1.0, 0.0, 0.0];
    let before = dot4(cam.look_up, world_up);

    for _ in 0..60 {
        cam.apply_mouse_look_transport(0.0, 0.0, 1.0, false);
    }
    let after = dot4(cam.look_up, world_up);

    assert!(
        after > before + 0.4,
        "expected look-transport damping to recover tilt: before={before} after={after}"
    );
}

#[test]
fn look_transport_non_hidden_yaw_hard_locks_roll_after_pitch() {
    let mut cam = Camera4D::new();
    cam.apply_mouse_look_transport(0.0, -120.0, 0.002, false);

    Camera4D::rotate_axis_pair(&mut cam.look_up, &mut cam.look_right, 0.35);
    cam.reorthonormalize_look_frame();

    cam.apply_mouse_look_transport(90.0, 0.0, 0.002, false);

    let world_up = [0.0, 1.0, 0.0, 0.0];
    let mut desired_up = Camera4D::sub_projection(world_up, cam.look_forward);
    desired_up = Camera4D::sub_projection(desired_up, cam.look_side);
    desired_up = Camera4D::normalize_with_fallback(desired_up, world_up);
    let alignment = dot4(cam.look_up, desired_up);
    assert!(
        alignment > 0.9999,
        "non-hidden yaw should immediately lock roll after pitch: alignment={alignment}"
    );
}

#[test]
fn look_transport_pull_to_home_improves_alignment_with_plus_x() {
    let mut cam = Camera4D::new();
    cam.apply_mouse_look_transport(160.0, -90.0, 0.002, false);
    cam.apply_mouse_look_transport(120.0, 40.0, 0.002, true);
    let before = dot4(cam.look_direction_look_frame(), [1.0, 0.0, 0.0, 0.0]);
    for _ in 0..12 {
        cam.pull_toward_home_look_frame(1.0 / 60.0);
    }
    let after = dot4(cam.look_direction_look_frame(), [1.0, 0.0, 0.0, 0.0]);
    assert!(
        after > before,
        "home pull should improve +X alignment: before={before} after={after}"
    );
}

#[test]
fn look_transport_pull_to_nearest_3d_reduces_axis_agnostic_3d_error() {
    let mut cam = Camera4D::new();

    Camera4D::rotate_axis_pair(&mut cam.look_up, &mut cam.look_side, 0.95);
    Camera4D::rotate_axis_pair(&mut cam.look_forward, &mut cam.look_side, 0.55);
    Camera4D::rotate_axis_pair(&mut cam.look_right, &mut cam.look_side, -0.35);
    cam.reorthonormalize_look_frame();

    let before = nearest_3d_pose_error(&cam);
    for _ in 0..12 {
        cam.pull_toward_nearest_3d_look_frame(1.0 / 60.0);
    }
    let after = nearest_3d_pose_error(&cam);
    assert!(
        after < before,
        "3D pull should reduce axis-agnostic 3D error: before={before} after={after}"
    );
}

#[test]
fn angle_pull_to_nearest_3d_reduces_hidden_angles() {
    let mut cam = Camera4D::new();
    cam.xw_angle = 1.2;
    cam.zw_angle = -0.9;
    cam.yw_deviation = 0.8;
    let before = (
        cam.xw_angle.abs(),
        cam.zw_angle.abs(),
        cam.yw_deviation.abs(),
    );
    cam.pull_toward_nearest_3d_angles(0.2);
    let after = (
        cam.xw_angle.abs(),
        cam.zw_angle.abs(),
        cam.yw_deviation.abs(),
    );
    assert!(
        after.0 < before.0 && after.1 < before.1 && after.2 < before.2,
        "3D angle pull should reduce hidden angles: before={before:?} after={after:?}"
    );
}

#[test]
fn angle_pull_to_home_reduces_all_angles() {
    let mut cam = Camera4D::new();
    cam.yaw = 0.9;
    cam.pitch = -0.5;
    cam.xw_angle = -1.1;
    cam.zw_angle = 0.7;
    cam.yw_deviation = -0.6;
    let before = (
        cam.yaw.abs(),
        cam.pitch.abs(),
        cam.xw_angle.abs(),
        cam.zw_angle.abs(),
        cam.yw_deviation.abs(),
    );
    cam.pull_toward_home_angles(0.2);
    let after = (
        cam.yaw.abs(),
        cam.pitch.abs(),
        cam.xw_angle.abs(),
        cam.zw_angle.abs(),
        cam.yw_deviation.abs(),
    );
    assert!(
        after.0 < before.0
            && after.1 < before.1
            && after.2 < before.2
            && after.3 < before.3
            && after.4 < before.4,
        "home angle pull should reduce all angles: before={before:?} after={after:?}"
    );
}

#[test]
fn rotor_mode_preserves_orthonormal_look_frame() {
    let mut cam = Camera4D::new();
    cam.apply_mouse_look_rotor(120.0, -80.0, 0.003, false, false);
    cam.apply_mouse_look_rotor(-90.0, 55.0, 0.002, true, false);
    cam.apply_mouse_look_rotor(75.0, 40.0, 0.0025, true, true);

    let (right, up, view_z, view_w) = cam.view_basis_look_frame();
    let basis = [right, up, view_z, view_w];
    for i in 0..4 {
        let norm = len4(basis[i]);
        assert!(
            (norm - 1.0).abs() < 1e-3,
            "basis vector {i} should stay unit length, got {norm}"
        );
        for j in (i + 1)..4 {
            let d = dot4(basis[i], basis[j]);
            assert!(
                d.abs() < 1e-3,
                "basis vectors {i},{j} should remain orthogonal, dot={d}"
            );
        }
    }
}

#[test]
fn look_transport_stays_orthonormal_under_mixed_inputs() {
    let mut cam = Camera4D::new();

    for i in 0..5_000 {
        let t = i as f32;
        let dx = (t * 0.037).sin() * 140.0 + (t * 0.013).cos() * 60.0;
        let dy = (t * 0.031).cos() * 90.0 + (t * 0.019).sin() * 45.0;
        let hidden = i % 7 < 3;
        cam.apply_mouse_look_transport(dx, dy, 0.0015, hidden);

        let (right, up, view_z, view_w) = cam.view_basis_look_frame();
        let basis = [right, up, view_z, view_w];
        for a in 0..4 {
            let n = len4(basis[a]);
            assert!(
                (n - 1.0).abs() < 1e-3,
                "vector {a} drifted from unit length: {n}"
            );
            for b in (a + 1)..4 {
                let d = dot4(basis[a], basis[b]);
                assert!(
                    d.abs() < 1e-3,
                    "vectors {a} and {b} lost orthogonality: dot={d}"
                );
            }
        }
    }
}

#[test]
fn look_transport_center_ray_matches_view_matrix() {
    let mut cam = Camera4D::new();
    cam.apply_mouse_look_transport(180.0, -120.0, 0.0018, false);
    cam.apply_mouse_look_transport(-90.0, 60.0, 0.0021, true);
    cam.apply_mouse_look_transport(35.0, 40.0, 0.0019, false);

    let got = cam.look_direction_look_frame();
    let view = cam.view_matrix_look_frame();
    let look_view = [
        0.0,
        0.0,
        std::f32::consts::FRAC_1_SQRT_2,
        std::f32::consts::FRAC_1_SQRT_2,
    ];
    let mut expected = [0.0f32; 4];
    for world_axis in 0..4 {
        let mut value = 0.0f32;
        for view_axis in 0..4 {
            value += view[[view_axis, world_axis]] * look_view[view_axis];
        }
        expected[world_axis] = value;
    }
    let len = len4(expected);
    assert!(len > 1e-6, "expected non-zero center-ray direction");
    for v in &mut expected {
        *v /= len;
    }

    for axis in 0..4 {
        assert!(
            (got[axis] - expected[axis]).abs() < 1e-4,
            "center-ray mismatch axis {axis}: got={} expected={}",
            got[axis],
            expected[axis]
        );
    }
}

#[test]
fn angles_for_direction_upright_roundtrip() {
    // Test various angle combinations: compute look direction, then recover angles,
    // then verify the recovered angles produce the same look direction.
    let test_cases = [
        (0.0, 0.0, 0.0, 0.0),
        (0.5, 0.3, 0.0, 0.0),
        (-1.0, -0.5, 0.4, -0.3),
        (0.0, 0.0, 1.0, 0.0),
        (0.0, 0.0, 0.0, -1.0),
        (2.0, 0.7, -0.8, 0.6),
        (-0.3, -0.2, 0.9, -0.7),
    ];
    for (yaw, pitch, xw, zw) in test_cases {
        let mut cam = Camera4D::new();
        cam.yaw = yaw;
        cam.pitch = pitch;
        cam.xw_angle = xw;
        cam.zw_angle = zw;
        cam.enforce_upright_constraints();

        let dir = cam.look_direction_upright();
        let (ry, rp, rxw, rzw) = Camera4D::angles_for_direction_upright(dir);

        // Set recovered angles and check look direction matches
        let mut cam2 = Camera4D::new();
        cam2.yaw = ry;
        cam2.pitch = rp;
        cam2.xw_angle = rxw;
        cam2.zw_angle = rzw;
        cam2.enforce_upright_constraints();
        let dir2 = cam2.look_direction_upright();

        for axis in 0..4 {
            assert!(
                (dir[axis] - dir2[axis]).abs() < 1e-3,
                "roundtrip failed for angles ({yaw},{pitch},{xw},{zw}): \
                     original dir={dir:?}, recovered dir={dir2:?}, \
                     recovered angles=({ry},{rp},{rxw},{rzw})"
            );
        }
    }
}

#[test]
fn look_frame_sync_from_standard_preserves_view_matrix() {
    let mut cam = Camera4D::new();
    cam.position = [3.0, -2.5, 1.2, 0.7];
    cam.yaw = 0.47;
    cam.pitch = -0.33;
    cam.xw_angle = 0.61;
    cam.zw_angle = -0.28;
    cam.yw_deviation = 0.22;

    let standard = cam.view_matrix();
    cam.sync_look_frame_from_standard_rotation();
    let look = cam.view_matrix_look_frame();

    for row in 0..5 {
        for col in 0..5 {
            assert!(
                (standard[[row, col]] - look[[row, col]]).abs() < 1e-4,
                "view mismatch at ({row},{col}): standard={} look={}",
                standard[[row, col]],
                look[[row, col]]
            );
        }
    }
}
