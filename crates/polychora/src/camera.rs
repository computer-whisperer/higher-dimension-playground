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
const ORIENTATION_PULL_RATE_HOME: f32 = 10.0;
const ORIENTATION_PULL_RATE_3D: f32 = 7.0;
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

    fn unit_axis(axis: usize) -> [f32; 4] {
        let mut v = [0.0f32; 4];
        v[axis.min(3)] = 1.0;
        v
    }

    fn project_out_axis(dir: [f32; 4], axis: usize) -> [f32; 4] {
        let mut v = dir;
        v[axis.min(3)] = 0.0;
        v
    }

    fn complement_axes(drop_axis: usize) -> [usize; 3] {
        let mut axes = [0usize; 3];
        let mut write = 0usize;
        for axis in 0..4 {
            if axis == drop_axis {
                continue;
            }
            axes[write] = axis;
            write += 1;
        }
        axes
    }

    fn cross_in_hyperplane(a: [f32; 4], b: [f32; 4], drop_axis: usize) -> [f32; 4] {
        let [i0, i1, i2] = Self::complement_axes(drop_axis);
        let av = [a[i0], a[i1], a[i2]];
        let bv = [b[i0], b[i1], b[i2]];
        let cv = [
            av[1] * bv[2] - av[2] * bv[1],
            av[2] * bv[0] - av[0] * bv[2],
            av[0] * bv[1] - av[1] * bv[0],
        ];
        let mut out = [0.0f32; 4];
        out[i0] = cv[0];
        out[i1] = cv[1];
        out[i2] = cv[2];
        out
    }

    fn hyperplane_fallback_axis(
        drop_axis: usize,
        avoid0: [f32; 4],
        avoid1: Option<[f32; 4]>,
    ) -> [f32; 4] {
        let axes = Self::complement_axes(drop_axis);
        let mut best = [0.0f32; 4];
        let mut best_len_sq = -1.0f32;

        for axis in axes {
            let mut candidate = Self::unit_axis(axis);
            candidate = Self::sub_projection(candidate, avoid0);
            if let Some(avoid) = avoid1 {
                candidate = Self::sub_projection(candidate, avoid);
            }
            let len_sq = Self::dot4(candidate, candidate);
            if len_sq > best_len_sq {
                best_len_sq = len_sq;
                best = candidate;
            }
        }

        Self::normalize_with_fallback(best, Self::unit_axis(axes[0]))
    }

    fn pull_alpha(rate: f32, dt: f32) -> f32 {
        1.0 - (-rate * dt.max(0.0)).exp()
    }

    fn pull_wrapped_toward(current: &mut f32, target: f32, alpha: f32) {
        let delta = wrap_angle(target - *current);
        *current = wrap_angle(*current + delta * alpha);
    }

    fn pull_pitch_toward(current: &mut f32, target: f32, alpha: f32) {
        *current += (target - *current) * alpha;
        let max_pitch = 81.0_f32.to_radians();
        *current = (*current).clamp(-max_pitch, max_pitch);
    }

    fn blend_look_frame_toward(
        &mut self,
        target_right: [f32; 4],
        target_up: [f32; 4],
        target_forward: [f32; 4],
        target_side: [f32; 4],
        alpha: f32,
    ) {
        let alpha = alpha.clamp(0.0, 1.0);
        if alpha <= 0.0 {
            return;
        }
        for i in 0..4 {
            self.look_right[i] = self.look_right[i] * (1.0 - alpha) + target_right[i] * alpha;
            self.look_up[i] = self.look_up[i] * (1.0 - alpha) + target_up[i] * alpha;
            self.look_forward[i] = self.look_forward[i] * (1.0 - alpha) + target_forward[i] * alpha;
            self.look_side[i] = self.look_side[i] * (1.0 - alpha) + target_side[i] * alpha;
        }
        self.reorthonormalize_look_frame();
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
        let mut xzw = [
            self.look_forward[0],
            self.look_forward[2],
            self.look_forward[3],
        ];
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
        let vertical_delta = -dy * sensitivity;
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
        if hidden_mode {
            // Hidden vertical maps to a direct ZW-like turn in baseline orientation.
            // This gives LOOK-TR a first-order hidden-plane axis like UPRIGHT's mod+mouse-Y.
            Self::rotate_axis_pair(&mut self.look_right, &mut self.look_side, vertical_delta);
        } else {
            // Normal vertical keeps classic visual pitch.
            Self::rotate_axis_pair(&mut self.look_forward, &mut self.look_up, vertical_delta);
        }

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

    /// Smoothly pull look-frame orientation toward the canonical +X 3D home pose.
    pub fn pull_toward_home_look_frame(&mut self, dt: f32) {
        let alpha = Self::pull_alpha(ORIENTATION_PULL_RATE_HOME, dt);
        self.blend_look_frame_toward(
            [0.0, 0.0, 1.0, 0.0],
            [0.0, 1.0, 0.0, 0.0],
            [1.0, 0.0, 0.0, 0.0],
            [0.0, 0.0, 0.0, 1.0],
            alpha,
        );
        self.stabilize_look_up_axis(LOOK_TRANSPORT_YAW_UPRIGHT_LOCK);
    }

    /// Smoothly pull look-frame orientation toward the nearest 3D-native pose:
    /// choose the hidden axis from current side dominance, then project the rest of
    /// the frame into the complementary 3D hyperplane.
    pub fn pull_toward_nearest_3d_look_frame(&mut self, dt: f32) {
        let alpha = Self::pull_alpha(ORIENTATION_PULL_RATE_3D, dt);
        let mut hidden_axis = 0usize;
        let mut hidden_abs = self.look_side[0].abs();
        for axis in 1..4 {
            let a = self.look_side[axis].abs();
            if a > hidden_abs {
                hidden_abs = a;
                hidden_axis = axis;
            }
        }
        let mut target_side = Self::unit_axis(hidden_axis);
        if self.look_side[hidden_axis] < 0.0 {
            for v in &mut target_side {
                *v = -*v;
            }
        }

        let forward_fallback = Self::hyperplane_fallback_axis(hidden_axis, self.look_up, None);
        let target_forward = Self::normalize_with_fallback(
            Self::project_out_axis(self.look_forward, hidden_axis),
            forward_fallback,
        );

        let mut target_up = Self::project_out_axis(self.look_up, hidden_axis);
        target_up = Self::sub_projection(target_up, target_forward);
        let up_fallback = Self::hyperplane_fallback_axis(hidden_axis, target_forward, None);
        target_up = Self::normalize_with_fallback(target_up, up_fallback);

        let mut target_right = Self::project_out_axis(self.look_right, hidden_axis);
        target_right = Self::sub_projection(target_right, target_forward);
        target_right = Self::sub_projection(target_right, target_up);
        let right_fallback = Self::normalize_with_fallback(
            Self::cross_in_hyperplane(target_forward, target_up, hidden_axis),
            Self::hyperplane_fallback_axis(hidden_axis, target_forward, Some(target_up)),
        );
        target_right = Self::normalize_with_fallback(target_right, right_fallback);

        target_up = Self::sub_projection(target_up, target_forward);
        target_up = Self::sub_projection(target_up, target_right);
        let up_final_fallback =
            Self::hyperplane_fallback_axis(hidden_axis, target_forward, Some(target_right));
        target_up = Self::normalize_with_fallback(target_up, up_final_fallback);

        self.blend_look_frame_toward(target_right, target_up, target_forward, target_side, alpha);
        self.stabilize_look_up_axis(LOOK_TRANSPORT_UPRIGHT_DAMPING);
    }

    /// Smoothly pull angle-parameterized orientation toward canonical +X home.
    pub fn pull_toward_home_angles(&mut self, dt: f32) {
        let alpha = Self::pull_alpha(ORIENTATION_PULL_RATE_HOME, dt);
        Self::pull_wrapped_toward(&mut self.yaw, 0.0, alpha);
        Self::pull_pitch_toward(&mut self.pitch, 0.0, alpha);
        Self::pull_wrapped_toward(&mut self.xw_angle, 0.0, alpha);
        Self::pull_wrapped_toward(&mut self.zw_angle, 0.0, alpha);
        Self::pull_wrapped_toward(&mut self.yw_deviation, 0.0, alpha);
    }

    /// Smoothly pull angle-parameterized orientation toward the nearest 3D-native pose.
    /// For angle modes, this means removing hidden-dimension rotations while preserving
    /// current yaw/pitch heading.
    pub fn pull_toward_nearest_3d_angles(&mut self, dt: f32) {
        let alpha = Self::pull_alpha(ORIENTATION_PULL_RATE_3D, dt);
        Self::pull_wrapped_toward(&mut self.xw_angle, 0.0, alpha);
        Self::pull_wrapped_toward(&mut self.zw_angle, 0.0, alpha);
        Self::pull_wrapped_toward(&mut self.yw_deviation, 0.0, alpha);
    }

    /// Smoothly pull angles toward specific target values (used by look-at).
    pub fn pull_toward_target_angles(
        &mut self,
        target_yaw: f32,
        target_pitch: f32,
        target_xw: f32,
        target_zw: f32,
        dt: f32,
    ) -> bool {
        let alpha = Self::pull_alpha(ORIENTATION_PULL_RATE_HOME, dt);
        Self::pull_wrapped_toward(&mut self.yaw, target_yaw, alpha);
        Self::pull_pitch_toward(&mut self.pitch, target_pitch, alpha);
        Self::pull_wrapped_toward(&mut self.xw_angle, target_xw, alpha);
        Self::pull_wrapped_toward(&mut self.zw_angle, target_zw, alpha);

        let yaw_err = wrap_angle(self.yaw - target_yaw).abs();
        let pitch_err = (self.pitch - target_pitch).abs();
        let xw_err = wrap_angle(self.xw_angle - target_xw).abs();
        let zw_err = wrap_angle(self.zw_angle - target_zw).abs();
        yaw_err < 0.01 && pitch_err < 0.01 && xw_err < 0.01 && zw_err < 0.01
    }

    /// Compute the upright-mode angles (yaw, pitch, xw_angle, zw_angle) that make
    /// the camera look in the given world-space direction.
    ///
    /// Analytical decomposition of the rotation chain:
    /// R = Pitch(p) * Yaw(y) * ZW(zw+pi/4) * XW(xw-pi/2)
    /// d = R^T * [0, 0, s2, s2], so R * d = [0, 0, s2, s2].
    pub fn angles_for_direction_upright(target_dir: [f32; 4]) -> (f32, f32, f32, f32) {
        let d = Self::normalize_dir(target_dir);
        let s2 = std::f32::consts::FRAC_1_SQRT_2;

        // R * d = Pitch(p) * Yaw(y) * ZW(a_zw) * XW(a_xw) * d = [0, 0, s2, s2]
        // where a_xw = xw_angle - PI/2, a_zw = zw_angle + PI/4.
        //
        // Step-by-step:
        //   u = XW(a_xw) * d:
        //     u[0] = d[0]*cos(a_xw) + d[3]*sin(a_xw)
        //     u[1] = d[1], u[2] = d[2]
        //     u[3] = -d[0]*sin(a_xw) + d[3]*cos(a_xw)
        //
        //   v = ZW(a_zw) * u:
        //     v[0] = u[0], v[1] = d[1]
        //     v[2] = d[2]*cos(a_zw) + u[3]*sin(a_zw)
        //     v[3] = -d[2]*sin(a_zw) + u[3]*cos(a_zw)
        //
        //   w = Yaw(y) * v:
        //     w[0] = v[0]*cos(y) + v[2]*sin(y)
        //     w[1] = d[1], w[2] = -v[0]*sin(y) + v[2]*cos(y)
        //     w[3] = v[3]
        //
        //   result = Pitch(p) * w:
        //     result[0] = w[0] = 0
        //     result[1] = -w[2]*sin(p) + d[1]*cos(p) = 0
        //     result[2] = w[2]*cos(p) + d[1]*sin(p) = s2
        //     result[3] = v[3] = s2
        //
        // Constraints: v[3] = s2, w[0] = 0.
        // From w[0] = 0: yaw y = atan2(-v[0], v[2])
        // From result[1] = 0: p = atan2(d[1], w[2])
        //
        // v[3] = s2 gives: -d[2]*sin(a_zw) + u[3]*cos(a_zw) = s2
        // where u[3] = -d[0]*sin(a_xw) + d[3]*cos(a_xw)
        //
        // We have 2 unknowns (a_xw, a_zw) and 1 constraint. The extra DOF
        // means multiple solutions exist. We choose the one that minimizes
        // a_xw^2 + a_zw^2 (smallest hidden-dim rotations).
        //
        // Parameterize: a_xw is free, a_zw is determined from v[3]=s2.
        // Given a_xw: u[3] = sqrt(d[0]^2+d[3]^2)*cos(a_xw+atan2(d[0],d[3]))
        // Then: v[3] = sqrt(d[2]^2+u[3]^2)*cos(a_zw+atan2(d[2],u[3])) = s2
        // So: a_zw = acos(s2/sqrt(d[2]^2+u[3]^2)) - atan2(d[2],u[3])
        //
        // Search over a_xw to minimize a_xw^2 + a_zw^2.
        // Since this is a 1D search, sample a few candidates and pick the best.

        let r_xw = (d[0] * d[0] + d[3] * d[3]).sqrt();
        let phi_xw = d[0].atan2(d[3]); // atan2(d[0], d[3])

        let best_solution = |a_xw: f32| -> Option<(f32, f32, f32, f32, f32)> {
            let u3 = r_xw * (a_xw + phi_xw).cos();
            let r_zw = (d[2] * d[2] + u3 * u3).sqrt();
            if r_zw < s2 - 1e-6 {
                return None; // Can't reach v[3] = s2
            }
            let ratio = (s2 / r_zw).clamp(-1.0, 1.0);
            let phi_zw = d[2].atan2(u3);
            // Two solutions: a_zw = +-acos(ratio) - phi_zw
            let base = ratio.acos();
            let a_zw_1 = wrap_angle(base - phi_zw);
            let a_zw_2 = wrap_angle(-base - phi_zw);

            // Pick the a_zw with smaller total cost
            let cost1 = a_xw * a_xw + a_zw_1 * a_zw_1;
            let cost2 = a_xw * a_xw + a_zw_2 * a_zw_2;
            let (a_zw, cost) = if cost1 <= cost2 {
                (a_zw_1, cost1)
            } else {
                (a_zw_2, cost2)
            };

            // Compute the remaining angles
            let u0 = d[0] * a_xw.cos() + d[3] * a_xw.sin();
            let u3_actual = -d[0] * a_xw.sin() + d[3] * a_xw.cos();
            let v2 = d[2] * a_zw.cos() + u3_actual * a_zw.sin();
            let v0 = u0;

            let y = (-v0).atan2(v2);

            // w[2] = -v0*sin(y) + v2*cos(y) = sqrt(v0^2 + v2^2) [positive by construction]
            let w2 = v0.hypot(v2);
            let p = d[1].atan2(w2);

            Some((y, p, a_xw, a_zw, cost))
        };

        // Sample a_xw values and find the one with minimum cost
        let n_samples = 64;
        let mut best: Option<(f32, f32, f32, f32, f32)> = None;

        for i in 0..n_samples {
            let a_xw = -PI + 2.0 * PI * (i as f32 + 0.5) / n_samples as f32;
            if let Some(sol) = best_solution(a_xw) {
                if best.is_none() || sol.4 < best.unwrap().4 {
                    best = Some(sol);
                }
            }
        }

        // Refine the best a_xw with golden section search
        if let Some((_, _, best_axw, _, _)) = best {
            let step = 2.0 * PI / n_samples as f32;
            let mut lo = best_axw - step;
            let mut hi = best_axw + step;
            let gr = 0.5 * (5.0_f32.sqrt() - 1.0); // golden ratio

            for _ in 0..30 {
                let mid1 = hi - gr * (hi - lo);
                let mid2 = lo + gr * (hi - lo);
                let c1 = best_solution(mid1).map_or(f32::INFINITY, |s| s.4);
                let c2 = best_solution(mid2).map_or(f32::INFINITY, |s| s.4);
                if c1 < c2 {
                    hi = mid2;
                } else {
                    lo = mid1;
                }
            }

            let final_axw = 0.5 * (lo + hi);
            if let Some((y, p, _, a_zw, _)) = best_solution(final_axw) {
                // Convert from effective angles back to camera angles
                let xw_angle = wrap_angle(final_axw - UPRIGHT_XW_ZERO_OFFSET);
                let zw_angle = wrap_angle(a_zw - UPRIGHT_ZW_ZERO_OFFSET);
                let max_pitch = 81.0_f32.to_radians();
                return (y, p.clamp(-max_pitch, max_pitch), xw_angle, zw_angle);
            }
        }

        // Fallback: no valid solution found (shouldn't happen for valid directions)
        (0.0, 0.0, 0.0, 0.0)
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
}
