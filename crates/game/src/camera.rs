use higher_dimension_playground::matrix_operations::{
    rotation_matrix_one_angle, translate_matrix_4d,
};
use ndarray::Array2;
use std::f32::consts::PI;

fn wrap_angle(a: f32) -> f32 {
    let mut a = a % (2.0 * PI);
    if a > PI {
        a -= 2.0 * PI;
    } else if a < -PI {
        a += 2.0 * PI;
    }
    a
}

const GRAVITY: f32 = 20.0;
const JUMP_SPEED: f32 = 8.0;
pub const PLAYER_HEIGHT: f32 = 1.7;
pub const PLAYER_RADIUS_XZW: f32 = 0.30;

#[derive(Clone, Copy, PartialEq, Eq)]
pub enum AngleTarget {
    Yaw,
    Pitch,
    XwAngle,
    ZwAngle,
    YwDeviation,
}

pub struct Camera4D {
    pub position: [f32; 4],
    pub yaw: f32,
    pub pitch: f32,
    pub xw_angle: f32,
    pub zw_angle: f32,
    pub yw_deviation: f32,
    pub is_flying: bool,
    pub is_grounded: bool,
    pub velocity_y: f32,
}

impl Camera4D {
    fn rotation_matrix(&self) -> Array2<f32> {
        let m = rotation_matrix_one_angle(5, 1, 3, self.yw_deviation); // YW (outermost)
        let m = m.dot(&rotation_matrix_one_angle(5, 2, 3, self.zw_angle)); // ZW
        let m = m.dot(&rotation_matrix_one_angle(5, 0, 3, self.xw_angle)); // XW
        let m = m.dot(&rotation_matrix_one_angle(5, 2, 1, self.pitch)); // ZY (pitch)
        m.dot(&rotation_matrix_one_angle(5, 0, 2, self.yaw)) // XZ (yaw, innermost)
    }

    /// Upright composition order for intuitive controls:
    /// XW first, then yaw, then pitch (applied right-to-left to vectors).
    fn rotation_matrix_upright(&self) -> Array2<f32> {
        let m = rotation_matrix_one_angle(5, 2, 1, self.pitch); // ZY (outermost)
        let m = m.dot(&rotation_matrix_one_angle(5, 0, 2, self.yaw)); // XZ
        m.dot(&rotation_matrix_one_angle(5, 0, 3, self.xw_angle)) // XW (innermost)
    }

    /// Convert a direction from view-space basis to world-space.
    fn world_direction_from_view_with_rotation(
        rotation: &Array2<f32>,
        view_dir: [f32; 4],
    ) -> [f32; 4] {
        let mut world = [0.0f32; 4];
        for world_axis in 0..4 {
            for view_axis in 0..4 {
                world[world_axis] += rotation[[view_axis, world_axis]] * view_dir[view_axis];
            }
        }
        world
    }

    /// Convert a direction from view-space basis to world-space.
    fn world_direction_from_view(&self, view_dir: [f32; 4]) -> [f32; 4] {
        let rotation = self.rotation_matrix();
        Self::world_direction_from_view_with_rotation(&rotation, view_dir)
    }

    fn normalize_dir(dir: [f32; 4]) -> [f32; 4] {
        let len_sq = dir[0] * dir[0] + dir[1] * dir[1] + dir[2] * dir[2] + dir[3] * dir[3];
        if len_sq <= 1e-8 {
            return dir;
        }
        let inv_len = len_sq.sqrt().recip();
        [
            dir[0] * inv_len,
            dir[1] * inv_len,
            dir[2] * inv_len,
            dir[3] * inv_len,
        ]
    }

    pub fn new() -> Self {
        Camera4D {
            position: [0.0, 0.0, -8.0, -4.0],
            yaw: PI * 0.125,
            pitch: 0.0,
            xw_angle: PI * 0.125,
            zw_angle: 0.0,
            yw_deviation: 0.0,
            is_flying: true,
            is_grounded: false,
            velocity_y: 0.0,
        }
    }

    pub fn adjust_angle(&mut self, target: AngleTarget, delta: f32) {
        match target {
            AngleTarget::Yaw => {
                self.yaw += delta;
                self.yaw = wrap_angle(self.yaw);
            }
            AngleTarget::Pitch => {
                self.pitch += delta;
                let max_pitch = 81.0_f32.to_radians();
                self.pitch = self.pitch.clamp(-max_pitch, max_pitch);
            }
            AngleTarget::XwAngle => {
                self.xw_angle += delta;
                self.xw_angle = wrap_angle(self.xw_angle);
            }
            AngleTarget::ZwAngle => {
                self.zw_angle += delta;
                self.zw_angle = wrap_angle(self.zw_angle);
            }
            AngleTarget::YwDeviation => self.yw_deviation += delta,
        }
    }

    pub fn apply_mouse_look_on(
        &mut self,
        dx: f32,
        dy: f32,
        sensitivity: f32,
        h_target: AngleTarget,
        v_target: AngleTarget,
    ) {
        self.adjust_angle(h_target, -dx * sensitivity);
        self.adjust_angle(v_target, -dy * sensitivity);
    }

    pub fn view_matrix(&self) -> Array2<f32> {
        let [px, py, pz, pw] = self.position;
        let m = self.rotation_matrix();
        let m = m.dot(&translate_matrix_4d(-px, -py, -pz, -pw));
        m
    }

    pub fn view_matrix_upright(&self) -> Array2<f32> {
        let [px, py, pz, pw] = self.position;
        let m = self.rotation_matrix_upright();
        let m = m.dot(&translate_matrix_4d(-px, -py, -pz, -pw));
        m
    }

    pub fn view_basis(&self) -> ([f32; 4], [f32; 4], [f32; 4], [f32; 4]) {
        let rotation = self.rotation_matrix();
        let right = Self::world_direction_from_view_with_rotation(&rotation, [1.0, 0.0, 0.0, 0.0]);
        let up = Self::world_direction_from_view_with_rotation(&rotation, [0.0, 1.0, 0.0, 0.0]);
        let view_z = Self::world_direction_from_view_with_rotation(&rotation, [0.0, 0.0, 1.0, 0.0]);
        let view_w = Self::world_direction_from_view_with_rotation(&rotation, [0.0, 0.0, 0.0, 1.0]);
        (right, up, view_z, view_w)
    }

    pub fn view_basis_upright(&self) -> ([f32; 4], [f32; 4], [f32; 4], [f32; 4]) {
        let rotation = self.rotation_matrix_upright();
        let right = Self::world_direction_from_view_with_rotation(&rotation, [1.0, 0.0, 0.0, 0.0]);
        let up = Self::world_direction_from_view_with_rotation(&rotation, [0.0, 1.0, 0.0, 0.0]);
        let view_z = Self::world_direction_from_view_with_rotation(&rotation, [0.0, 0.0, 1.0, 0.0]);
        let view_w = Self::world_direction_from_view_with_rotation(&rotation, [0.0, 0.0, 0.0, 1.0]);
        (right, up, view_z, view_w)
    }

    /// World-space look direction for the center of the current Z/W view.
    pub fn look_direction(&self) -> [f32; 4] {
        let rotation = self.rotation_matrix();
        let look = Self::world_direction_from_view_with_rotation(
            &rotation,
            [
                0.0,
                0.0,
                std::f32::consts::FRAC_1_SQRT_2,
                std::f32::consts::FRAC_1_SQRT_2,
            ],
        );
        Self::normalize_dir(look)
    }

    pub fn look_direction_upright(&self) -> [f32; 4] {
        let rotation = self.rotation_matrix_upright();
        let look = Self::world_direction_from_view_with_rotation(
            &rotation,
            [
                0.0,
                0.0,
                std::f32::consts::FRAC_1_SQRT_2,
                std::f32::consts::FRAC_1_SQRT_2,
            ],
        );
        Self::normalize_dir(look)
    }

    pub fn auto_level(&mut self, dt: f32) {
        self.yw_deviation *= (-6.0 * dt).exp();
        if self.yw_deviation.abs() < 0.001 {
            self.yw_deviation = 0.0;
        }
    }

    pub fn reset_orientation(&mut self) {
        self.yaw = PI * 0.125;
        self.pitch = 0.0;
        self.xw_angle = PI * 0.125;
        self.zw_angle = 0.0;
        self.yw_deviation = 0.0;
    }

    /// Constrain camera orientation to an upright, no-roll/no-twist mode.
    pub fn enforce_upright_constraints(&mut self) {
        self.zw_angle = 0.0;
        self.yw_deviation = 0.0;
        let max_pitch = 81.0_f32.to_radians();
        self.pitch = self.pitch.clamp(-max_pitch, max_pitch);
    }

    /// Check if the effective up direction is flipped (view Y doesn't match world Y).
    /// This happens when XW/ZW rotations move past ±90° and cause pitch to appear inverted.
    pub fn is_y_inverted(&self) -> bool {
        // The view rotation row for Y (row 1) tells us what world direction maps to view-Y.
        // If the Y component of that row is negative, world-up maps to view-down.
        let m = self.rotation_matrix();
        // Row 1 (Y view axis), column 1 (Y world axis)
        m[(1, 1)] < 0.0
    }

    pub fn is_y_inverted_upright(&self) -> bool {
        let m = self.rotation_matrix_upright();
        m[(1, 1)] < 0.0
    }

    pub fn apply_movement(
        &mut self,
        forward: f32,
        strafe: f32,
        vertical: f32,
        w_axis: f32,
        dt: f32,
        speed: f32,
    ) {
        let rotation = self.rotation_matrix();
        let right = Self::world_direction_from_view_with_rotation(&rotation, [1.0, 0.0, 0.0, 0.0]);
        let up = Self::world_direction_from_view_with_rotation(&rotation, [0.0, 1.0, 0.0, 0.0]);
        let view_z = Self::world_direction_from_view_with_rotation(&rotation, [0.0, 0.0, 1.0, 0.0]);
        let view_w = Self::world_direction_from_view_with_rotation(&rotation, [0.0, 0.0, 0.0, 1.0]);

        let movement_scale = speed * dt;
        let mut delta = [0.0f32; 4];
        for axis in 0..4 {
            delta[axis] = (forward * view_z[axis] + strafe * right[axis] + w_axis * view_w[axis])
                * movement_scale;
            if self.is_flying {
                delta[axis] += vertical * up[axis] * movement_scale;
            }
        }
        if !self.is_flying {
            // Keep gravity-mode locomotion on the world XZW hyperplane.
            delta[1] = 0.0;
        }
        for axis in 0..4 {
            self.position[axis] += delta[axis];
        }
    }

    /// Friendly movement scheme:
    /// - forward follows the center look ray (between view-Z and view-W)
    /// - Q/E moves on the orthogonal hidden-dimension side axis in the ZW plane
    /// - input is normalized to avoid diagonal speed boost
    /// - in gravity mode, movement is re-normalized on XZW so pitch does not slow down walking
    pub fn apply_movement_upright(
        &mut self,
        forward: f32,
        strafe: f32,
        vertical: f32,
        w_axis: f32,
        dt: f32,
        speed: f32,
    ) {
        let rotation = self.rotation_matrix_upright();
        let right = Self::world_direction_from_view_with_rotation(&rotation, [1.0, 0.0, 0.0, 0.0]);
        let up = Self::world_direction_from_view_with_rotation(&rotation, [0.0, 1.0, 0.0, 0.0]);
        let view_z = Self::world_direction_from_view_with_rotation(&rotation, [0.0, 0.0, 1.0, 0.0]);
        let view_w = Self::world_direction_from_view_with_rotation(&rotation, [0.0, 0.0, 0.0, 1.0]);

        let center_forward = Self::normalize_dir([
            view_z[0] + view_w[0],
            view_z[1] + view_w[1],
            view_z[2] + view_w[2],
            view_z[3] + view_w[3],
        ]);
        let side_w = Self::normalize_dir([
            view_w[0] - view_z[0],
            view_w[1] - view_z[1],
            view_w[2] - view_z[2],
            view_w[3] - view_z[3],
        ]);

        let vertical_input = if self.is_flying { vertical } else { 0.0 };
        let input_len = (forward * forward
            + strafe * strafe
            + w_axis * w_axis
            + vertical_input * vertical_input)
            .sqrt();
        if input_len <= 1e-8 {
            return;
        }
        let input_mag = input_len.min(1.0);
        let input_scale = if input_len > 1.0 {
            input_len.recip()
        } else {
            1.0
        };
        let forward_i = forward * input_scale;
        let strafe_i = strafe * input_scale;
        let w_i = w_axis * input_scale;
        let vertical_i = vertical_input * input_scale;

        let mut wish = [0.0f32; 4];
        for axis in 0..4 {
            wish[axis] = forward_i * center_forward[axis]
                + strafe_i * right[axis]
                + w_i * side_w[axis]
                + vertical_i * up[axis];
        }

        if !self.is_flying {
            wish[1] = 0.0;
        }

        let wish_len_sq =
            wish[0] * wish[0] + wish[1] * wish[1] + wish[2] * wish[2] + wish[3] * wish[3];
        if wish_len_sq <= 1e-8 {
            return;
        }
        let movement_scale = speed * dt * input_mag * wish_len_sq.sqrt().recip();
        for axis in 0..4 {
            self.position[axis] += wish[axis] * movement_scale;
        }
    }

    pub fn toggle_flying(&mut self) {
        self.is_flying = !self.is_flying;
        if self.is_flying {
            self.is_grounded = false;
        }
        self.velocity_y = 0.0;
    }

    pub fn jump(&mut self) {
        if self.is_grounded {
            self.velocity_y = JUMP_SPEED;
            self.is_grounded = false;
        }
    }

    pub fn update_physics(&mut self, dt: f32) {
        if self.is_flying {
            return;
        }
        self.velocity_y -= GRAVITY * dt;
        self.position[1] += self.velocity_y * dt;
    }
}

#[cfg(test)]
mod tests {
    use super::*;

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
    fn movement_uses_full_view_rotation_basis() {
        let mut cam = Camera4D::new();
        cam.position = [0.0, 0.0, 0.0, 0.0];
        cam.yaw = 0.37;
        cam.pitch = -0.22;
        cam.xw_angle = 0.61;
        cam.zw_angle = -0.48;
        cam.yw_deviation = 0.19;
        cam.is_flying = true;

        let expected = cam.world_direction_from_view([0.0, 0.0, 1.0, 0.0]);
        cam.apply_movement(1.0, 0.0, 0.0, 0.0, 1.0, 1.0);
        for axis in 0..4 {
            assert!((cam.position[axis] - expected[axis]).abs() < 1e-4);
        }
    }

    #[test]
    fn upright_constraints_remove_roll_twist_and_keep_y_non_inverted() {
        let mut cam = Camera4D::new();
        cam.pitch = 2.0;
        cam.zw_angle = 0.73;
        cam.yw_deviation = -1.11;
        cam.enforce_upright_constraints();

        assert_eq!(cam.zw_angle, 0.0);
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

        let expected = cam.look_direction_upright();
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
            let along_right = delta[0] * right[0]
                + delta[1] * right[1]
                + delta[2] * right[2]
                + delta[3] * right[3];
            assert!(
                along_right > 0.0,
                "upright yaw should not invert at xw={:.2} deg (delta·right={:.6e})",
                cam.xw_angle.to_degrees(),
                along_right
            );
        }
    }
}
