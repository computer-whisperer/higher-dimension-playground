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
        let m = rotation_matrix_one_angle(5, 1, 3, self.yw_deviation); // YW (outermost)
        let m = m.dot(&rotation_matrix_one_angle(5, 2, 3, self.zw_angle)); // ZW
        let m = m.dot(&rotation_matrix_one_angle(5, 0, 3, self.xw_angle)); // XW
        let m = m.dot(&rotation_matrix_one_angle(5, 2, 1, self.pitch)); // ZY (pitch)
        let m = m.dot(&rotation_matrix_one_angle(5, 0, 2, self.yaw)); // XZ (yaw, innermost)
        let m = m.dot(&translate_matrix_4d(-px, -py, -pz, -pw));
        m
    }

    /// World-space look direction for the center of the current Z/W view.
    pub fn look_direction(&self) -> [f32; 4] {
        let view = self.view_matrix();
        let look_view = [
            0.0,
            0.0,
            std::f32::consts::FRAC_1_SQRT_2,
            std::f32::consts::FRAC_1_SQRT_2,
        ];
        let mut look = [0.0f32; 4];

        // For pure directions, inverse(view_rotation) = transpose(view_rotation).
        for world_axis in 0..4 {
            for view_axis in 0..4 {
                look[world_axis] += view[[view_axis, world_axis]] * look_view[view_axis];
            }
        }

        let len_sq = look[0] * look[0] + look[1] * look[1] + look[2] * look[2] + look[3] * look[3];
        if len_sq > 1e-8 {
            let inv_len = len_sq.sqrt().recip();
            for axis in &mut look {
                *axis *= inv_len;
            }
        }
        look
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

    /// Check if the effective up direction is flipped (view Y doesn't match world Y).
    /// This happens when XW/ZW rotations move past ±90° and cause pitch to appear inverted.
    pub fn is_y_inverted(&self) -> bool {
        // The view matrix row for Y (row 1) tells us what world direction maps to view-Y.
        // If the Y component of that row is negative, world-up maps to view-down.
        let m = rotation_matrix_one_angle(5, 1, 3, self.yw_deviation);
        let m = m.dot(&rotation_matrix_one_angle(5, 2, 3, self.zw_angle));
        let m = m.dot(&rotation_matrix_one_angle(5, 0, 3, self.xw_angle));
        let m = m.dot(&rotation_matrix_one_angle(5, 2, 1, self.pitch));
        // Row 1 (Y view axis), column 1 (Y world axis)
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
        let fwd_x = -self.yaw.sin();
        let fwd_z = self.yaw.cos();

        self.position[0] += (forward * fwd_x + strafe * fwd_z) * speed * dt;
        if self.is_flying {
            self.position[1] += vertical * speed * dt;
        }
        self.position[2] += (forward * fwd_z - strafe * fwd_x) * speed * dt;
        self.position[3] += w_axis * speed * dt;
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
    fn mouse_up_looks_up_when_not_y_inverted() {
        let mut cam = Camera4D::new();
        cam.yaw = 0.0;
        cam.pitch = 0.0;
        cam.xw_angle = 0.0;
        cam.zw_angle = 0.0;
        cam.yw_deviation = 0.0;
        assert!(!cam.is_y_inverted(), "baseline camera should not be Y-inverted");

        let before_y = cam.look_direction()[1];
        cam.apply_mouse_look_on(0.0, -40.0, 0.01, AngleTarget::Yaw, AngleTarget::Pitch);
        let after_y = cam.look_direction()[1];
        assert!(
            after_y > before_y,
            "mouse-up should increase look.y when not Y-inverted: before={before_y:.4} after={after_y:.4}"
        );
    }

    #[test]
    fn pitch_response_can_reverse_at_some_orientations() {
        let mut cam = Camera4D::new();

        let mut found_mouse_up_moves_up = false;
        let mut found_mouse_up_moves_down = false;

        // Scan a coarse orientation grid and measure vertical response
        // from the standard pitch control.
        for yaw_step in -3..=3 {
            for pitch_step in -2..=2 {
                for xw_step in -3..=3 {
                    for zw_step in -3..=3 {
                        cam.yaw = yaw_step as f32 * (std::f32::consts::PI / 8.0);
                        cam.pitch = pitch_step as f32 * (std::f32::consts::PI / 10.0);
                        cam.xw_angle = xw_step as f32 * (std::f32::consts::PI / 8.0);
                        cam.zw_angle = zw_step as f32 * (std::f32::consts::PI / 8.0);
                        cam.yw_deviation = 0.0;

                        let before_y = cam.look_direction()[1];
                        cam.apply_mouse_look_on(
                            0.0,
                            -10.0,
                            0.01,
                            AngleTarget::Yaw,
                            AngleTarget::Pitch,
                        );
                        let after_y = cam.look_direction()[1];
                        let dy = after_y - before_y;

                        if dy > 1e-4 {
                            found_mouse_up_moves_up = true;
                        } else if dy < -1e-4 {
                            found_mouse_up_moves_down = true;
                        }
                    }
                }
            }
        }

        assert!(
            found_mouse_up_moves_up,
            "expected at least one orientation where mouse-up pitches upward"
        );
        assert!(
            found_mouse_up_moves_down,
            "expected at least one orientation where mouse-up pitches downward"
        );
    }
}
