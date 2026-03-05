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
const TRANSPORT_MODERATE_STABILIZATION: f32 = 0.12;
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
        self.apply_mouse_look_transport_with_modifiers(dx, dy, sensitivity, hidden_mode, false);
    }

    pub fn apply_mouse_look_transport_with_modifiers(
        &mut self,
        dx: f32,
        dy: f32,
        sensitivity: f32,
        hidden_mode: bool,
        forward_mode: bool,
    ) {
        let yaw_delta = dx * sensitivity;
        let vertical_delta = -dy * sensitivity;
        if forward_mode {
            // Mouse-forward LOOK-TR mode drives the two remaining look-frame planes:
            // right/up (roll-like) and up/side (hidden-elevation-like).
            Self::rotate_axis_pair(&mut self.look_right, &mut self.look_up, yaw_delta);
            Self::rotate_axis_pair(&mut self.look_up, &mut self.look_side, vertical_delta);
            self.reorthonormalize_look_frame();
            // Intentionally skip upright stabilization while forward-modifying.
            return;
        }

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

    /// Transport Uniform (TR-UNI): full 4D rotations in all modes, uniform stabilization.
    /// No XZW constraints — pole speed issue cannot occur. Stabilization keeps horizon feel.
    pub fn apply_mouse_look_transport_uniform(
        &mut self,
        dx: f32,
        dy: f32,
        sensitivity: f32,
        hidden_mode: bool,
        forward_mode: bool,
    ) {
        let h_delta = dx * sensitivity;
        let v_delta = -dy * sensitivity;

        if forward_mode {
            Self::rotate_axis_pair(&mut self.look_right, &mut self.look_up, h_delta);
            Self::rotate_axis_pair(&mut self.look_up, &mut self.look_side, v_delta);
        } else if hidden_mode {
            Self::rotate_axis_pair(&mut self.look_forward, &mut self.look_side, h_delta);
            Self::rotate_axis_pair(&mut self.look_right, &mut self.look_side, v_delta);
        } else {
            Self::rotate_axis_pair(&mut self.look_forward, &mut self.look_right, h_delta);
            Self::rotate_axis_pair(&mut self.look_forward, &mut self.look_up, v_delta);
        }

        self.reorthonormalize_look_frame();
        self.stabilize_look_up_axis(TRANSPORT_MODERATE_STABILIZATION);
    }

    /// Transport Decoupled (TR-DEC): full 4D rotations in all modes, stabilization only
    /// in default mode. Modifier modes give precise 4D control without the system fighting you.
    pub fn apply_mouse_look_transport_decoupled(
        &mut self,
        dx: f32,
        dy: f32,
        sensitivity: f32,
        hidden_mode: bool,
        forward_mode: bool,
    ) {
        let h_delta = dx * sensitivity;
        let v_delta = -dy * sensitivity;

        if forward_mode {
            Self::rotate_axis_pair(&mut self.look_right, &mut self.look_up, h_delta);
            Self::rotate_axis_pair(&mut self.look_up, &mut self.look_side, v_delta);
        } else if hidden_mode {
            Self::rotate_axis_pair(&mut self.look_forward, &mut self.look_side, h_delta);
            Self::rotate_axis_pair(&mut self.look_right, &mut self.look_side, v_delta);
        } else {
            Self::rotate_axis_pair(&mut self.look_forward, &mut self.look_right, h_delta);
            Self::rotate_axis_pair(&mut self.look_forward, &mut self.look_up, v_delta);
        }

        self.reorthonormalize_look_frame();
        if !hidden_mode && !forward_mode {
            self.stabilize_look_up_axis(TRANSPORT_MODERATE_STABILIZATION);
        }
    }

    /// Transport Scaled (TR-SCL): XZW-constrained yaw like LOOK-TR, but with cos(elevation)
    /// sensitivity scaling to fix pole speed. Other rotations are full 4D.
    pub fn apply_mouse_look_transport_scaled(
        &mut self,
        dx: f32,
        dy: f32,
        sensitivity: f32,
        hidden_mode: bool,
        forward_mode: bool,
    ) {
        let yaw_delta = dx * sensitivity;
        let vertical_delta = -dy * sensitivity;

        if forward_mode {
            Self::rotate_axis_pair(&mut self.look_right, &mut self.look_up, yaw_delta);
            Self::rotate_axis_pair(&mut self.look_up, &mut self.look_side, vertical_delta);
            self.reorthonormalize_look_frame();
            return;
        }

        // Scale yaw by cos(elevation) — the XZW magnitude of look_forward.
        // Near the poles this approaches 0, compensating for the amplification
        // that XZW-constrained rotation causes when the XZW projection is small.
        let xzw_mag = (self.look_forward[0].powi(2)
            + self.look_forward[2].powi(2)
            + self.look_forward[3].powi(2))
        .sqrt();
        let scaled_yaw = yaw_delta * xzw_mag;

        let forward_y_before = self.look_forward[1];
        if hidden_mode {
            Self::rotate_axis_pair_xzw(&mut self.look_forward, &mut self.look_side, scaled_yaw);
        } else {
            Self::rotate_axis_pair_xzw(&mut self.look_forward, &mut self.look_right, scaled_yaw);
        }
        self.renormalize_forward_preserve_y(forward_y_before);

        if hidden_mode {
            Self::rotate_axis_pair(&mut self.look_right, &mut self.look_side, vertical_delta);
        } else {
            Self::rotate_axis_pair(&mut self.look_forward, &mut self.look_up, vertical_delta);
        }

        self.reorthonormalize_look_frame();
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

    /// Smoothly pull look-frame orientation to face a target direction.
    /// Constructs a target frame from the direction, keeping the side axis
    /// as close to the current one as possible and the up axis close to world +Y.
    /// Returns true when converged.
    pub fn pull_toward_target_direction_look_frame(
        &mut self,
        target_dir: [f32; 4],
        dt: f32,
    ) -> bool {
        let alpha = Self::pull_alpha(ORIENTATION_PULL_RATE_HOME, dt);
        let target_forward = Self::normalize_with_fallback(target_dir, self.look_forward);

        // Side: project current side out of target_forward, normalize
        let mut target_side = Self::sub_projection(self.look_side, target_forward);
        target_side = Self::normalize_with_fallback(target_side, [0.0, 0.0, 0.0, 1.0]);

        // Up: world +Y projected away from forward and side
        let world_up = [0.0, 1.0, 0.0, 0.0];
        let mut target_up = Self::sub_projection(world_up, target_forward);
        target_up = Self::sub_projection(target_up, target_side);
        target_up = Self::normalize_with_fallback(target_up, self.look_up);

        // Right: whatever is left in the orthogonal complement
        let mut target_right = Self::sub_projection(self.look_right, target_forward);
        target_right = Self::sub_projection(target_right, target_side);
        target_right = Self::sub_projection(target_right, target_up);
        target_right = Self::normalize_with_fallback(target_right, [0.0, 0.0, 1.0, 0.0]);

        self.blend_look_frame_toward(target_right, target_up, target_forward, target_side, alpha);
        self.stabilize_look_up_axis(LOOK_TRANSPORT_YAW_UPRIGHT_LOCK);

        // Check convergence
        let dot = Self::dot4(self.look_forward, target_forward);
        dot > 0.9999
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
mod tests;
