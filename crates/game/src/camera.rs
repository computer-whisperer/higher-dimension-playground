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
const LOOK_TRANSPORT_UPRIGHT_DAMPING: f32 = 0.06;
const LOOK_TRANSPORT_YAW_UPRIGHT_LOCK: f32 = 1.0;
const UPRIGHT_YAW_ZERO_OFFSET: f32 = 0.0;
const UPRIGHT_ZW_ZERO_OFFSET: f32 = std::f32::consts::FRAC_PI_4;
const UPRIGHT_XW_ZERO_OFFSET: f32 = -std::f32::consts::FRAC_PI_2;

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
    look_right: [f32; 4],
    look_up: [f32; 4],
    look_forward: [f32; 4],
    look_side: [f32; 4],
    look_frame_initialized: bool,
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
    /// XW first, then ZW, then yaw, then pitch (applied right-to-left to vectors).
    fn rotation_matrix_upright(&self) -> Array2<f32> {
        // Re-center upright zero angles to a simple axis-aligned forward (+X).
        let yaw = self.yaw + UPRIGHT_YAW_ZERO_OFFSET;
        let zw = self.zw_angle + UPRIGHT_ZW_ZERO_OFFSET;
        let xw = self.xw_angle + UPRIGHT_XW_ZERO_OFFSET;

        let m = rotation_matrix_one_angle(5, 2, 1, self.pitch); // ZY (outermost)
        let m = m.dot(&rotation_matrix_one_angle(5, 0, 2, yaw)); // XZ
        let m = m.dot(&rotation_matrix_one_angle(5, 2, 3, zw)); // ZW
        m.dot(&rotation_matrix_one_angle(5, 0, 3, xw)) // XW (innermost)
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

    /// Normalize a direction after projecting it onto the world XZW hyperplane.
    fn normalize_xzw(dir: [f32; 4]) -> [f32; 4] {
        let planar = [dir[0], 0.0, dir[2], dir[3]];
        let len_sq = planar[0] * planar[0] + planar[2] * planar[2] + planar[3] * planar[3];
        if len_sq <= 1e-8 {
            return [0.0, 0.0, 0.0, 0.0];
        }
        let inv_len = len_sq.sqrt().recip();
        [
            planar[0] * inv_len,
            0.0,
            planar[2] * inv_len,
            planar[3] * inv_len,
        ]
    }

    fn dot4(a: [f32; 4], b: [f32; 4]) -> f32 {
        a[0] * b[0] + a[1] * b[1] + a[2] * b[2] + a[3] * b[3]
    }

    fn sub_projection(v: [f32; 4], axis: [f32; 4]) -> [f32; 4] {
        let d = Self::dot4(v, axis);
        [
            v[0] - d * axis[0],
            v[1] - d * axis[1],
            v[2] - d * axis[2],
            v[3] - d * axis[3],
        ]
    }

    fn normalize_with_fallback(dir: [f32; 4], fallback: [f32; 4]) -> [f32; 4] {
        let len_sq = Self::dot4(dir, dir);
        if len_sq <= 1e-8 {
            return fallback;
        }
        let inv_len = len_sq.sqrt().recip();
        [
            dir[0] * inv_len,
            dir[1] * inv_len,
            dir[2] * inv_len,
            dir[3] * inv_len,
        ]
    }

    fn rotate_axis_pair(a: &mut [f32; 4], b: &mut [f32; 4], angle: f32) {
        let c = angle.cos();
        let s = angle.sin();
        let old_a = *a;
        let old_b = *b;
        for i in 0..4 {
            a[i] = c * old_a[i] + s * old_b[i];
            b[i] = -s * old_a[i] + c * old_b[i];
        }
    }

    fn rotate_axis_pair_xzw(a: &mut [f32; 4], b: &mut [f32; 4], angle: f32) {
        let c = angle.cos();
        let s = angle.sin();
        let old_a = *a;
        let old_b = *b;
        for i in [0usize, 2usize, 3usize] {
            a[i] = c * old_a[i] + s * old_b[i];
            b[i] = -s * old_a[i] + c * old_b[i];
        }
    }

    fn renormalize_forward_preserve_y(&mut self, target_y: f32) {
        let y = target_y.clamp(-1.0, 1.0);
        let mut xzw = [self.look_forward[0], self.look_forward[2], self.look_forward[3]];
        let len_sq = xzw[0] * xzw[0] + xzw[1] * xzw[1] + xzw[2] * xzw[2];
        if len_sq <= 1e-12 {
            return;
        }
        let inv_len = len_sq.sqrt().recip();
        xzw[0] *= inv_len;
        xzw[1] *= inv_len;
        xzw[2] *= inv_len;
        let horiz_len = (1.0 - y * y).max(0.0).sqrt();
        self.look_forward[0] = xzw[0] * horiz_len;
        self.look_forward[1] = y;
        self.look_forward[2] = xzw[1] * horiz_len;
        self.look_forward[3] = xzw[2] * horiz_len;
    }

    fn reorthonormalize_look_frame(&mut self) {
        self.look_forward = Self::normalize_with_fallback(self.look_forward, [1.0, 0.0, 0.0, 0.0]);

        let mut side = Self::sub_projection(self.look_side, self.look_forward);
        side = Self::normalize_with_fallback(side, [0.0, 0.0, 0.0, 1.0]);
        self.look_side = side;

        let mut up = Self::sub_projection(self.look_up, self.look_forward);
        up = Self::sub_projection(up, self.look_side);
        let world_up = [0.0, 1.0, 0.0, 0.0];
        let mut up_fallback = Self::sub_projection(world_up, self.look_forward);
        up_fallback = Self::sub_projection(up_fallback, self.look_side);
        up_fallback = Self::normalize_with_fallback(up_fallback, [0.0, 1.0, 0.0, 0.0]);
        self.look_up = Self::normalize_with_fallback(up, up_fallback);

        let mut right = Self::sub_projection(self.look_right, self.look_forward);
        right = Self::sub_projection(right, self.look_side);
        right = Self::sub_projection(right, self.look_up);
        right = Self::normalize_with_fallback(right, [0.0, 0.0, 1.0, 0.0]);
        self.look_right = right;
    }

    fn stabilize_look_up_axis(&mut self, blend: f32) {
        let world_up = [0.0, 1.0, 0.0, 0.0];
        let mut desired_up = Self::sub_projection(world_up, self.look_forward);
        desired_up = Self::sub_projection(desired_up, self.look_side);
        let desired_len_sq = Self::dot4(desired_up, desired_up);
        if desired_len_sq <= 1e-8 {
            return;
        }
        let inv_len = desired_len_sq.sqrt().recip();
        for v in &mut desired_up {
            *v *= inv_len;
        }

        let alpha = blend.clamp(0.0, 1.0);
        let blended = [
            self.look_up[0] * (1.0 - alpha) + desired_up[0] * alpha,
            self.look_up[1] * (1.0 - alpha) + desired_up[1] * alpha,
            self.look_up[2] * (1.0 - alpha) + desired_up[2] * alpha,
            self.look_up[3] * (1.0 - alpha) + desired_up[3] * alpha,
        ];
        self.look_up = Self::normalize_with_fallback(blended, desired_up);
        self.reorthonormalize_look_frame();

        // Keep the visual up hemisphere consistent with world +Y.
        if Self::dot4(self.look_up, world_up) < 0.0 {
            for i in 0..4 {
                self.look_up[i] = -self.look_up[i];
                self.look_right[i] = -self.look_right[i];
            }
        }
    }

    fn look_view_zw_axes(&self) -> ([f32; 4], [f32; 4]) {
        let inv_sqrt2 = std::f32::consts::FRAC_1_SQRT_2;
        let view_z = [
            (self.look_forward[0] - self.look_side[0]) * inv_sqrt2,
            (self.look_forward[1] - self.look_side[1]) * inv_sqrt2,
            (self.look_forward[2] - self.look_side[2]) * inv_sqrt2,
            (self.look_forward[3] - self.look_side[3]) * inv_sqrt2,
        ];
        let view_w = [
            (self.look_forward[0] + self.look_side[0]) * inv_sqrt2,
            (self.look_forward[1] + self.look_side[1]) * inv_sqrt2,
            (self.look_forward[2] + self.look_side[2]) * inv_sqrt2,
            (self.look_forward[3] + self.look_side[3]) * inv_sqrt2,
        ];
        (view_z, view_w)
    }

    fn sync_look_frame_from_rotation_matrix(&mut self, rotation: Array2<f32>) {
        self.look_right = [
            rotation[[0, 0]],
            rotation[[0, 1]],
            rotation[[0, 2]],
            rotation[[0, 3]],
        ];
        self.look_up = [
            rotation[[1, 0]],
            rotation[[1, 1]],
            rotation[[1, 2]],
            rotation[[1, 3]],
        ];
        let view_z = [
            rotation[[2, 0]],
            rotation[[2, 1]],
            rotation[[2, 2]],
            rotation[[2, 3]],
        ];
        let view_w = [
            rotation[[3, 0]],
            rotation[[3, 1]],
            rotation[[3, 2]],
            rotation[[3, 3]],
        ];
        self.look_forward = Self::normalize_with_fallback(
            [
                view_z[0] + view_w[0],
                view_z[1] + view_w[1],
                view_z[2] + view_w[2],
                view_z[3] + view_w[3],
            ],
            [1.0, 0.0, 0.0, 0.0],
        );
        self.look_side = Self::normalize_with_fallback(
            [
                view_w[0] - view_z[0],
                view_w[1] - view_z[1],
                view_w[2] - view_z[2],
                view_w[3] - view_z[3],
            ],
            [0.0, 0.0, 0.0, 1.0],
        );
        self.reorthonormalize_look_frame();
        self.look_frame_initialized = true;
    }

    pub fn sync_look_frame_from_standard_rotation(&mut self) {
        self.sync_look_frame_from_rotation_matrix(self.rotation_matrix());
    }

    pub fn sync_look_frame_from_upright_rotation(&mut self) {
        self.sync_look_frame_from_rotation_matrix(self.rotation_matrix_upright());
    }

    pub fn look_frame_initialized(&self) -> bool {
        self.look_frame_initialized
    }

    pub fn apply_mouse_look_transport(
        &mut self,
        dx: f32,
        dy: f32,
        sensitivity: f32,
        hidden_mode: bool,
    ) {
        let yaw_delta = dx * sensitivity;
        let pitch_delta = -dy * sensitivity;
        let forward_y_before_yaw = self.look_forward[1];

        if hidden_mode {
            // Keep yaw strictly on the world XZW hyperplane: do not mix Y into yaw turns.
            Self::rotate_axis_pair_xzw(&mut self.look_forward, &mut self.look_side, yaw_delta);
        } else {
            // Keep yaw strictly on the world XZW hyperplane: do not mix Y into yaw turns.
            Self::rotate_axis_pair_xzw(&mut self.look_forward, &mut self.look_right, yaw_delta);
        }
        // Preserve visual pitch across yaw so yaw never directly tilts toward/away from Y.
        self.renormalize_forward_preserve_y(forward_y_before_yaw);
        // Keep vertical mouse motion on the visual up axis in both modes; hidden mode
        // only changes the horizontal plane from forward/right to forward/side.
        Self::rotate_axis_pair(&mut self.look_forward, &mut self.look_up, pitch_delta);

        self.reorthonormalize_look_frame();
        // Regular yaw should feel horizon-locked; hidden turns keep the gentler damping.
        let upright_blend = if !hidden_mode && yaw_delta != 0.0 {
            LOOK_TRANSPORT_YAW_UPRIGHT_LOCK
        } else {
            LOOK_TRANSPORT_UPRIGHT_DAMPING
        };
        self.stabilize_look_up_axis(upright_blend);
    }

    pub fn apply_mouse_look_rotor(
        &mut self,
        dx: f32,
        dy: f32,
        sensitivity: f32,
        hidden_mode: bool,
        spin_mode: bool,
    ) {
        let h_delta = dx * sensitivity;
        let v_delta = -dy * sensitivity;

        if hidden_mode && spin_mode {
            Self::rotate_axis_pair(&mut self.look_right, &mut self.look_side, h_delta);
            Self::rotate_axis_pair(&mut self.look_up, &mut self.look_side, v_delta);
        } else if hidden_mode {
            Self::rotate_axis_pair(&mut self.look_forward, &mut self.look_side, h_delta);
            Self::rotate_axis_pair(&mut self.look_up, &mut self.look_side, v_delta);
        } else if spin_mode {
            Self::rotate_axis_pair(&mut self.look_right, &mut self.look_side, h_delta);
            Self::rotate_axis_pair(&mut self.look_forward, &mut self.look_up, v_delta);
        } else {
            Self::rotate_axis_pair(&mut self.look_forward, &mut self.look_right, h_delta);
            Self::rotate_axis_pair(&mut self.look_forward, &mut self.look_up, v_delta);
        }

        self.reorthonormalize_look_frame();
    }

    pub fn view_matrix_look_frame(&self) -> Array2<f32> {
        let [px, py, pz, pw] = self.position;
        let mut m = Array2::<f32>::eye(5);
        let (view_z, view_w) = self.look_view_zw_axes();
        for axis in 0..4 {
            m[[0, axis]] = self.look_right[axis];
            m[[1, axis]] = self.look_up[axis];
            m[[2, axis]] = view_z[axis];
            m[[3, axis]] = view_w[axis];
        }
        m.dot(&translate_matrix_4d(-px, -py, -pz, -pw))
    }

    pub fn view_basis_look_frame(&self) -> ([f32; 4], [f32; 4], [f32; 4], [f32; 4]) {
        let (view_z, view_w) = self.look_view_zw_axes();
        (self.look_right, self.look_up, view_z, view_w)
    }

    pub fn look_direction_look_frame(&self) -> [f32; 4] {
        self.look_forward
    }

    pub fn is_y_inverted_look_frame(&self) -> bool {
        Self::dot4(self.look_up, [0.0, 1.0, 0.0, 0.0]) < 0.0
    }

    pub fn apply_movement_look_frame(
        &mut self,
        forward: f32,
        strafe: f32,
        vertical: f32,
        w_axis: f32,
        dt: f32,
        speed: f32,
    ) {
        let right = Self::normalize_xzw(self.look_right);
        let center_forward = Self::normalize_xzw(self.look_forward);
        let side_w = Self::normalize_xzw(self.look_side);

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
            wish[axis] =
                forward_i * center_forward[axis] + strafe_i * right[axis] + w_i * side_w[axis];
        }
        wish[1] += vertical_i;

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

    pub fn new() -> Self {
        Camera4D {
            // Stand on top of the default flat-ground surface (voxel top at y=0).
            position: [0.0, PLAYER_HEIGHT, -8.0, -4.0],
            yaw: 0.0,
            pitch: 0.0,
            xw_angle: 0.0,
            zw_angle: 0.0,
            yw_deviation: 0.0,
            look_right: [0.0, 0.0, 1.0, 0.0],
            look_up: [0.0, 1.0, 0.0, 0.0],
            look_forward: [1.0, 0.0, 0.0, 0.0],
            look_side: [0.0, 0.0, 0.0, 1.0],
            look_frame_initialized: true,
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
            AngleTarget::YwDeviation => {
                self.yw_deviation += delta;
                self.yw_deviation = wrap_angle(self.yw_deviation);
            }
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
        self.yaw = 0.0;
        self.pitch = 0.0;
        self.xw_angle = 0.0;
        self.zw_angle = 0.0;
        self.yw_deviation = 0.0;
        self.look_right = [0.0, 0.0, 1.0, 0.0];
        self.look_up = [0.0, 1.0, 0.0, 0.0];
        self.look_forward = [1.0, 0.0, 0.0, 0.0];
        self.look_side = [0.0, 0.0, 0.0, 1.0];
        self.look_frame_initialized = true;
    }

    /// Constrain camera orientation to an upright, no-roll/no-twist mode.
    pub fn enforce_upright_constraints(&mut self) {
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
        let right = Self::normalize_xzw(Self::world_direction_from_view_with_rotation(
            &rotation,
            [1.0, 0.0, 0.0, 0.0],
        ));
        let view_z = Self::normalize_xzw(Self::world_direction_from_view_with_rotation(
            &rotation,
            [0.0, 0.0, 1.0, 0.0],
        ));
        let view_w = Self::normalize_xzw(Self::world_direction_from_view_with_rotation(
            &rotation,
            [0.0, 0.0, 0.0, 1.0],
        ));

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
            wish[axis] = forward_i * view_z[axis] + strafe_i * right[axis] + w_i * view_w[axis];
        }
        // Minecraft-like vertical controls: Space/Shift move only along world Y.
        wish[1] += vertical_i;

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

    /// Friendly movement scheme:
    /// - forward follows the center look ray (between view-Z and view-W)
    /// - Q/E moves on the orthogonal hidden-dimension side axis in the ZW plane
    /// - forward/strafe/QE are projected to world XZW (no Y drift from look pitch)
    /// - Space/Shift moves exclusively on world Y when flying
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
        let right = Self::normalize_xzw(Self::world_direction_from_view_with_rotation(
            &rotation,
            [1.0, 0.0, 0.0, 0.0],
        ));
        let view_z = Self::world_direction_from_view_with_rotation(&rotation, [0.0, 0.0, 1.0, 0.0]);
        let view_w = Self::world_direction_from_view_with_rotation(&rotation, [0.0, 0.0, 0.0, 1.0]);

        let center_forward = Self::normalize_xzw([
            view_z[0] + view_w[0],
            view_z[1] + view_w[1],
            view_z[2] + view_w[2],
            view_z[3] + view_w[3],
        ]);
        let side_w = Self::normalize_xzw([
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
            wish[axis] =
                forward_i * center_forward[axis] + strafe_i * right[axis] + w_i * side_w[axis];
        }
        // Minecraft-like vertical controls: Space/Shift move only along world Y.
        wish[1] += vertical_i;

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

    fn dot4(a: [f32; 4], b: [f32; 4]) -> f32 {
        a[0] * b[0] + a[1] * b[1] + a[2] * b[2] + a[3] * b[3]
    }

    fn len4(a: [f32; 4]) -> f32 {
        dot4(a, a).sqrt()
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
    fn look_transport_hidden_mode_vertical_uses_visual_pitch() {
        let mut cam = Camera4D::new();
        cam.apply_mouse_look_transport(0.0, -std::f32::consts::FRAC_PI_2, 1.0, true);
        let look = cam.look_direction_look_frame();
        assert!(look[1] > 0.9999, "expected +Y aim, got y={}", look[1]);
        assert!(
            look[3].abs() < 1e-4,
            "expected w≈0 for pure pitch, got w={}",
            look[3]
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
}
