use super::*;

impl Scene {
    fn trace_first_solid_voxel(
        &self,
        ray_origin: [f32; 4],
        ray_direction: [f32; 4],
        max_distance: f32,
    ) -> Option<VoxelRayHit> {
        let dir_len_sq = ray_direction[0] * ray_direction[0]
            + ray_direction[1] * ray_direction[1]
            + ray_direction[2] * ray_direction[2]
            + ray_direction[3] * ray_direction[3];
        if dir_len_sq <= 1e-8 || max_distance <= 0.0 {
            return None;
        }

        let inv_dir_len = dir_len_sq.sqrt().recip();
        let dir = [
            ray_direction[0] * inv_dir_len,
            ray_direction[1] * inv_dir_len,
            ray_direction[2] * inv_dir_len,
            ray_direction[3] * inv_dir_len,
        ];
        let origin = [
            ray_origin[0] + dir[0] * EDIT_RAY_EPSILON,
            ray_origin[1] + dir[1] * EDIT_RAY_EPSILON,
            ray_origin[2] + dir[2] * EDIT_RAY_EPSILON,
            ray_origin[3] + dir[3] * EDIT_RAY_EPSILON,
        ];

        let mut cell = [
            origin[0].floor() as i32,
            origin[1].floor() as i32,
            origin[2].floor() as i32,
            origin[3].floor() as i32,
        ];

        let mut step = [0i32; 4];
        let mut t_max = [f32::INFINITY; 4];
        let mut t_delta = [f32::INFINITY; 4];

        for axis in 0..4 {
            let d = dir[axis];
            if d > 1e-8 {
                step[axis] = 1;
                let next_boundary = cell[axis] as f32 + 1.0;
                t_max[axis] = (next_boundary - origin[axis]) / d;
                t_delta[axis] = 1.0 / d;
            } else if d < -1e-8 {
                step[axis] = -1;
                let next_boundary = cell[axis] as f32;
                t_max[axis] = (next_boundary - origin[axis]) / d;
                t_delta[axis] = -1.0 / d;
            }
        }

        let mut last_empty_voxel = None;
        let v0 = self.get_block_data(cell[0], cell[1], cell[2], cell[3]);
        if !v0.is_air() {
            return Some(VoxelRayHit {
                solid_voxel: cell,
                last_empty_voxel,
            });
        }
        last_empty_voxel = Some(cell);

        for _ in 0..EDIT_RAY_MAX_STEPS {
            let mut axis = 0usize;
            for candidate in 1..4 {
                if t_max[candidate] < t_max[axis] {
                    axis = candidate;
                }
            }

            let traversed_t = t_max[axis];
            if !traversed_t.is_finite() || traversed_t > max_distance {
                break;
            }

            cell[axis] += step[axis];
            t_max[axis] += t_delta[axis];

            let voxel = self.get_block_data(cell[0], cell[1], cell[2], cell[3]);
            if !voxel.is_air() {
                return Some(VoxelRayHit {
                    solid_voxel: cell,
                    last_empty_voxel,
                });
            }
            last_empty_voxel = Some(cell);
        }

        None
    }

    /// Remove the first solid voxel intersected by a camera ray.
    #[cfg(test)]
    pub fn remove_block_along_ray(
        &mut self,
        ray_origin: [f32; 4],
        ray_direction: [f32; 4],
        max_distance: f32,
    ) -> Option<[i32; 4]> {
        let hit = self.trace_first_solid_voxel(ray_origin, ray_direction, max_distance)?;
        let [x, y, z, w] = hit.solid_voxel;
        self.world_set_block(x, y, z, w, BlockData::AIR);
        Some(hit.solid_voxel)
    }

    /// Query edit ray targets without mutating the world.
    pub fn block_edit_targets(
        &self,
        ray_origin: [f32; 4],
        ray_direction: [f32; 4],
        max_distance: f32,
    ) -> BlockEditTargets {
        match self.trace_first_solid_voxel(ray_origin, ray_direction, max_distance) {
            Some(hit) => BlockEditTargets {
                hit_voxel: Some(hit.solid_voxel),
                place_voxel: hit.last_empty_voxel,
            },
            None => BlockEditTargets::default(),
        }
    }

    /// Fan-cast across the ZW viewing wedge and return the nearest solid
    /// voxel hit.  `view_z` and `view_w` are the camera's world-space Z and W
    /// basis vectors (obtained from the view basis).  The sweep mirrors the VTE
    /// shader: theta ranges from `PI/4 - viewAngle/2` to `PI/4 + viewAngle/2`
    /// where `viewAngle = (PI/2) / focal_length_zw`.
    pub fn fan_cast_nearest_block(
        &self,
        ray_origin: [f32; 4],
        view_z: [f32; 4],
        view_w: [f32; 4],
        focal_length_zw: f32,
        max_distance: f32,
        num_samples: usize,
    ) -> Option<[i32; 4]> {
        let pi = std::f32::consts::PI;
        let view_angle = (pi / 2.0) / focal_length_zw.max(0.01);
        let theta_min = pi / 4.0 - view_angle / 2.0;
        let theta_max = pi / 4.0 + view_angle / 2.0;

        let samples = num_samples.max(1);
        let mut best_voxel: Option<[i32; 4]> = None;
        let mut best_dist_sq = f32::INFINITY;

        for i in 0..samples {
            let t = if samples == 1 {
                0.5
            } else {
                i as f32 / (samples - 1) as f32
            };
            let theta = theta_min + t * (theta_max - theta_min);
            let cz = theta.cos();
            let sw = theta.sin();

            let dir = [
                cz * view_z[0] + sw * view_w[0],
                cz * view_z[1] + sw * view_w[1],
                cz * view_z[2] + sw * view_w[2],
                cz * view_z[3] + sw * view_w[3],
            ];

            if let Some(hit) = self.trace_first_solid_voxel(ray_origin, dir, max_distance) {
                let dx = hit.solid_voxel[0] as f32 + 0.5 - ray_origin[0];
                let dy = hit.solid_voxel[1] as f32 + 0.5 - ray_origin[1];
                let dz = hit.solid_voxel[2] as f32 + 0.5 - ray_origin[2];
                let dw = hit.solid_voxel[3] as f32 + 0.5 - ray_origin[3];
                let dist_sq = dx * dx + dy * dy + dz * dz + dw * dw;
                if dist_sq < best_dist_sq {
                    best_dist_sq = dist_sq;
                    best_voxel = Some(hit.solid_voxel);
                }
            }
        }

        best_voxel
    }

    /// Place a voxel in the last empty cell before the first solid hit.
    #[cfg(test)]
    pub fn place_block_along_ray(
        &mut self,
        ray_origin: [f32; 4],
        ray_direction: [f32; 4],
        max_distance: f32,
        material: BlockData,
    ) -> Option<[i32; 4]> {
        if material.is_air() {
            return None;
        }
        let hit = self.trace_first_solid_voxel(ray_origin, ray_direction, max_distance)?;
        let target = hit.last_empty_voxel?;
        let [x, y, z, w] = target;
        if !self.get_block_data(x, y, z, w).is_air() {
            return None;
        }
        self.world_set_block(x, y, z, w, material);
        Some(target)
    }
}
