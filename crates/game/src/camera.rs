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
const PLAYER_HEIGHT: f32 = 1.7;
const FLOOR_Y: f32 = -3.0;

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
        self.velocity_y = 0.0;
    }

    pub fn on_ground(&self) -> bool {
        self.position[1] <= FLOOR_Y + PLAYER_HEIGHT + 0.01
    }

    pub fn jump(&mut self) {
        if self.on_ground() {
            self.velocity_y = JUMP_SPEED;
        }
    }

    pub fn update_physics(&mut self, dt: f32) {
        if self.is_flying {
            return;
        }
        self.velocity_y -= GRAVITY * dt;
        self.position[1] += self.velocity_y * dt;
        let ground_y = FLOOR_Y + PLAYER_HEIGHT;
        if self.position[1] < ground_y {
            self.position[1] = ground_y;
            self.velocity_y = 0.0;
        }
    }
}
