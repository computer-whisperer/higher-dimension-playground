use super::*;

impl Scene {
    fn player_aabb(pos: [f32; 4]) -> ([f32; 4], [f32; 4]) {
        let min = [
            pos[0] - PLAYER_RADIUS_XZW,
            pos[1] - PLAYER_HEIGHT,
            pos[2] - PLAYER_RADIUS_XZW,
            pos[3] - PLAYER_RADIUS_XZW,
        ];
        let max = [
            pos[0] + PLAYER_RADIUS_XZW,
            pos[1],
            pos[2] + PLAYER_RADIUS_XZW,
            pos[3] + PLAYER_RADIUS_XZW,
        ];
        (min, max)
    }

    fn aabb_intersects_solid(&self, min: [f32; 4], max: [f32; 4]) -> bool {
        for axis in 0..4 {
            if let Some(lo) = self.world_bounds.min[axis] {
                if min[axis] < lo {
                    return true;
                }
            }
            if let Some(hi) = self.world_bounds.max[axis] {
                if max[axis] > hi {
                    return true;
                }
            }
        }

        let max_epsilon = 1e-4f32;
        let lo = [
            min[0].floor() as i32,
            min[1].floor() as i32,
            min[2].floor() as i32,
            min[3].floor() as i32,
        ];
        let hi = [
            (max[0] - max_epsilon).floor() as i32,
            (max[1] - max_epsilon).floor() as i32,
            (max[2] - max_epsilon).floor() as i32,
            (max[3] - max_epsilon).floor() as i32,
        ];
        if hi[0] < lo[0] || hi[1] < lo[1] || hi[2] < lo[2] || hi[3] < lo[3] {
            return false;
        }

        for x in lo[0]..=hi[0] {
            for y in lo[1]..=hi[1] {
                for z in lo[2]..=hi[2] {
                    for w in lo[3]..=hi[3] {
                        if !self.get_block_data(x, y, z, w).is_air() {
                            return true;
                        }
                    }
                }
            }
        }
        false
    }

    fn player_collides_at(&self, pos: [f32; 4]) -> bool {
        let (min, max) = Self::player_aabb(pos);
        self.aabb_intersects_solid(min, max)
    }

    fn depenetrate_upward(&self, pos: &mut [f32; 4]) {
        if !self.player_collides_at(*pos) {
            return;
        }
        for _ in 0..COLLISION_MAX_PUSHUP_STEPS {
            pos[1] += COLLISION_PUSHUP_STEP;
            if !self.player_collides_at(*pos) {
                return;
            }
        }
    }

    /// Resolve player movement against solid voxels.
    /// Returns `(resolved_position, grounded)`.
    pub fn resolve_player_collision(
        &self,
        old_pos: [f32; 4],
        attempted_pos: [f32; 4],
        velocity_y: &mut f32,
    ) -> ([f32; 4], bool) {
        let mut pos = old_pos;
        self.depenetrate_upward(&mut pos);

        for axis in [0usize, 2, 3, 1] {
            let target = attempted_pos[axis];
            if (target - pos[axis]).abs() <= 1e-6 {
                continue;
            }

            let mut candidate = pos;
            candidate[axis] = target;
            if !self.player_collides_at(candidate) {
                pos = candidate;
                continue;
            }

            let mut feasible = pos[axis];
            let mut blocked = target;
            for _ in 0..COLLISION_BINARY_STEPS {
                let mid = 0.5 * (feasible + blocked);
                let mut probe = pos;
                probe[axis] = mid;
                if self.player_collides_at(probe) {
                    blocked = mid;
                } else {
                    feasible = mid;
                }
            }
            pos[axis] = feasible;

            if axis == 1 {
                *velocity_y = 0.0;
            }
        }

        let mut ground_probe = pos;
        ground_probe[1] -= 0.02;
        let grounded = self.player_collides_at(ground_probe);
        if grounded && *velocity_y < 0.0 {
            *velocity_y = 0.0;
        }

        (pos, grounded)
    }
}
