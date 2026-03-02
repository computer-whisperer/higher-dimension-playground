use super::*;
use polychora::shared::region_tree::BvhRayHit;

impl Scene {
    /// Cast a ray into the world tree, returning the first solid block hit.
    fn raycast_solid(
        &self,
        ray_origin: [f32; 4],
        ray_direction: [f32; 4],
        max_distance: f32,
    ) -> Option<BvhRayHit> {
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

        let fp_origin = origin.map(ChunkCoord::from_num);
        let fp_dir = dir.map(ChunkCoord::from_num);
        let fp_max_t = ChunkCoord::from_num(max_distance);

        self.world_tree.raycast(fp_origin, fp_dir, fp_max_t)
    }

    /// Remove the first solid block intersected by a camera ray.
    /// Returns the block-aligned origin of the removed block.
    #[cfg(test)]
    pub fn remove_block_along_ray(
        &mut self,
        ray_origin: [f32; 4],
        ray_direction: [f32; 4],
        max_distance: f32,
    ) -> Option<[i32; 4]> {
        let hit = self.raycast_solid(ray_origin, ray_direction, max_distance)?;
        if hit.block.is_air() {
            return None;
        }
        let origin_i32: [i32; 4] = std::array::from_fn(|i| hit.bounds.min[i].to_num::<i32>());
        let size = 1i32 << hit.block.scale_exp.max(0);
        for dx in 0..size {
            for dy in 0..size {
                for dz in 0..size {
                    for dw in 0..size {
                        self.world_set_block(
                            origin_i32[0] + dx,
                            origin_i32[1] + dy,
                            origin_i32[2] + dz,
                            origin_i32[3] + dw,
                            BlockData::AIR,
                        );
                    }
                }
            }
        }
        Some(origin_i32)
    }

    /// Query edit ray targets without mutating the world.
    pub fn block_edit_targets(
        &self,
        ray_origin: [f32; 4],
        ray_direction: [f32; 4],
        max_distance: f32,
        placement_scale_exp: i8,
    ) -> BlockEditTargets {
        let hit = match self.raycast_solid(ray_origin, ray_direction, max_distance) {
            Some(h) if !h.block.is_air() => h,
            _ => return BlockEditTargets::default(),
        };

        let hit_target = ScaleAwareBlockTarget {
            origin: hit.bounds.min,
            scale_exp: hit.block.scale_exp,
        };
        let place_target = if hit.face_sign != 0 {
            Some(compute_placement_target(
                &hit_target,
                hit.hit_point,
                hit.face_axis,
                hit.face_sign,
                placement_scale_exp,
            ))
        } else {
            None
        };
        BlockEditTargets {
            hit: Some(hit_target),
            hit_block: Some(hit.block),
            place: place_target,
            face_axis: hit.face_axis,
            face_sign: hit.face_sign,
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

            if let Some(hit) = self.raycast_solid(ray_origin, dir, max_distance) {
                if hit.block.is_air() {
                    continue;
                }
                let origin: [i32; 4] =
                    std::array::from_fn(|i| hit.bounds.min[i].to_num::<i32>());
                let size = 1i32 << hit.block.scale_exp.max(0);
                let half = size as f32 * 0.5;
                let dx = origin[0] as f32 + half - ray_origin[0];
                let dy = origin[1] as f32 + half - ray_origin[1];
                let dz = origin[2] as f32 + half - ray_origin[2];
                let dw = origin[3] as f32 + half - ray_origin[3];
                let dist_sq = dx * dx + dy * dy + dz * dz + dw * dw;
                if dist_sq < best_dist_sq {
                    best_dist_sq = dist_sq;
                    best_voxel = Some(origin);
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
        let targets =
            self.block_edit_targets(ray_origin, ray_direction, max_distance, material.scale_exp);
        let place = targets.place?;
        let place_i32 = place.origin_i32();
        let size = 1i32 << place.scale_exp.max(0);
        // Verify all cells of the placement target are air
        for dx in 0..size {
            for dy in 0..size {
                for dz in 0..size {
                    for dw in 0..size {
                        if !self
                            .get_block_data(
                                place_i32[0] + dx,
                                place_i32[1] + dy,
                                place_i32[2] + dz,
                                place_i32[3] + dw,
                            )
                            .is_air()
                        {
                            return None;
                        }
                    }
                }
            }
        }
        // Fill all cells
        for dx in 0..size {
            for dy in 0..size {
                for dz in 0..size {
                    for dw in 0..size {
                        self.world_set_block(
                            place_i32[0] + dx,
                            place_i32[1] + dy,
                            place_i32[2] + dz,
                            place_i32[3] + dw,
                            material.clone(),
                        );
                    }
                }
            }
        }
        Some(place_i32)
    }
}

/// Compute scale-aware placement target adjacent to a hit block face.
fn compute_placement_target(
    hit: &ScaleAwareBlockTarget,
    hit_point: [ChunkCoord; 4],
    face_axis: u8,
    face_sign: i8,
    placement_scale_exp: i8,
) -> ScaleAwareBlockTarget {
    let hit_size = hit.size();
    let place_step = step_for_scale(placement_scale_exp);
    let mut origin = [ChunkCoord::ZERO; 4];
    for axis in 0..4 {
        if axis == face_axis as usize {
            origin[axis] = if face_sign < 0 {
                hit.origin[axis] - place_step
            } else {
                hit.origin[axis] + hit_size
            };
        } else {
            // Snap hit_point to placement grid
            let bits = hit_point[axis].to_bits();
            let step_bits = place_step.to_bits();
            origin[axis] =
                ChunkCoord::from_bits(bits.div_euclid(step_bits) * step_bits);
        }
    }
    ScaleAwareBlockTarget {
        origin,
        scale_exp: placement_scale_exp,
    }
}
