use std::f32::consts::PI;
use ndarray::Array2;
use higher_dimension_playground::matrix_operations::{translate_matrix_4d, rotation_matrix_one_angle};

pub struct Camera4D {
    pub position: [f32; 4],
    pub yaw: f32,
    pub pitch: f32,
}

impl Camera4D {
    pub fn new() -> Self {
        Camera4D {
            position: [0.0, 0.0, -8.0, -4.0],
            yaw: PI * 0.125,
            pitch: 0.0,
        }
    }

    pub fn view_matrix(&self) -> Array2<f32> {
        let [px, py, pz, pw] = self.position;
        let m = rotation_matrix_one_angle(5, 0, 3, PI * 0.125);
        let m = m.dot(&rotation_matrix_one_angle(5, 2, 1, self.pitch));
        let m = m.dot(&rotation_matrix_one_angle(5, 0, 2, self.yaw));
        let m = m.dot(&translate_matrix_4d(-px, -py, -pz, -pw));
        m
    }

    pub fn apply_movement(&mut self, forward: f32, strafe: f32, vertical: f32, w_axis: f32, dt: f32, speed: f32) {
        let fwd_x = -self.yaw.sin();
        let fwd_z = self.yaw.cos();

        self.position[0] += (forward * fwd_x + strafe * fwd_z) * speed * dt;
        self.position[1] += vertical * speed * dt;
        self.position[2] += (forward * fwd_z - strafe * fwd_x) * speed * dt;
        self.position[3] += w_axis * speed * dt;
    }

    pub fn apply_mouse_look(&mut self, dx: f32, dy: f32, sensitivity: f32) {
        self.yaw -= dx * sensitivity;
        self.pitch -= dy * sensitivity;

        let max_pitch = 81.0_f32.to_radians();
        self.pitch = self.pitch.clamp(-max_pitch, max_pitch);
    }
}
