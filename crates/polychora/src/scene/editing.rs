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

        let fp_origin = origin.map(ChunkCoord::from_num);
        let fp_dir = dir.map(ChunkCoord::from_num);
        let fp_max_t = ChunkCoord::from_num(max_distance);

        let hit = self.world_tree.raycast(fp_origin, fp_dir, fp_max_t)?;

        let solid_voxel: [i32; 4] = std::array::from_fn(|i| hit.bounds.min[i].to_num::<i32>());

        let t_f32: f32 = hit.t.to_num::<f32>();
        let last_empty_voxel = if t_f32 > EDIT_RAY_EPSILON {
            let step_back = t_f32 - EDIT_RAY_EPSILON;
            Some(std::array::from_fn(|i| {
                (origin[i] + dir[i] * step_back).floor() as i32
            }))
        } else {
            None
        };

        Some(VoxelRayHit {
            solid_voxel,
            last_empty_voxel,
            hit_block: hit.block,
        })
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
        let hit = self.trace_first_solid_voxel(ray_origin, ray_direction, max_distance)?;
        let size = 1i32 << hit.hit_block.scale_exp.max(0);
        let origin: [i32; 4] =
            std::array::from_fn(|i| hit.solid_voxel[i].div_euclid(size) * size);
        // Clear all cells of the block
        for dx in 0..size {
            for dy in 0..size {
                for dz in 0..size {
                    for dw in 0..size {
                        self.world_set_block(
                            origin[0] + dx,
                            origin[1] + dy,
                            origin[2] + dz,
                            origin[3] + dw,
                            BlockData::AIR,
                        );
                    }
                }
            }
        }
        Some(origin)
    }

    /// Query edit ray targets without mutating the world.
    pub fn block_edit_targets(
        &self,
        ray_origin: [f32; 4],
        ray_direction: [f32; 4],
        max_distance: f32,
        placement_scale_exp: i8,
    ) -> BlockEditTargets {
        match self.trace_first_solid_voxel(ray_origin, ray_direction, max_distance) {
            Some(hit) => {
                let size = 1i32 << hit.hit_block.scale_exp.max(0);
                let hit_origin: [i32; 4] =
                    std::array::from_fn(|i| hit.solid_voxel[i].div_euclid(size) * size);
                let hit_target = ScaleAwareBlockTarget {
                    origin: hit_origin,
                    scale_exp: hit.hit_block.scale_exp,
                };
                let place_target = hit.last_empty_voxel.and_then(|empty_cell| {
                    compute_placement_target(&hit_target, empty_cell, placement_scale_exp)
                });
                BlockEditTargets {
                    hit: Some(hit_target),
                    hit_block: Some(hit.hit_block),
                    place: place_target,
                }
            }
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
                let size = 1i32 << hit.hit_block.scale_exp.max(0);
                let origin: [i32; 4] =
                    std::array::from_fn(|i| hit.solid_voxel[i].div_euclid(size) * size);
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
        let targets = self.block_edit_targets(ray_origin, ray_direction, max_distance, material.scale_exp);
        let place = targets.place?;
        // Verify all cells of the placement target are air
        let size = place.size();
        for dx in 0..size {
            for dy in 0..size {
                for dz in 0..size {
                    for dw in 0..size {
                        if !self
                            .get_block_data(
                                place.origin[0] + dx,
                                place.origin[1] + dy,
                                place.origin[2] + dz,
                                place.origin[3] + dw,
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
                            place.origin[0] + dx,
                            place.origin[1] + dy,
                            place.origin[2] + dz,
                            place.origin[3] + dw,
                            material.clone(),
                        );
                    }
                }
            }
        }
        Some(place.origin)
    }
}

/// Compute scale-aware placement target adjacent to a hit block face.
fn compute_placement_target(
    hit: &ScaleAwareBlockTarget,
    empty_cell: [i32; 4],
    placement_scale_exp: i8,
) -> Option<ScaleAwareBlockTarget> {
    let hit_size = hit.size();
    let place_size = 1i32 << placement_scale_exp.max(0);
    let hit_min = hit.origin;
    let hit_max: [i32; 4] = std::array::from_fn(|i| hit_min[i] + hit_size);

    // Find the face axis: the axis where empty_cell is outside the hit block's extent.
    let mut face_axis = None;
    let mut face_negative = false;
    for axis in 0..4 {
        if empty_cell[axis] < hit_min[axis] {
            if face_axis.is_some() {
                // Ambiguous — empty cell is outside on multiple axes (corner).
                // Fall back to the first found axis.
            } else {
                face_axis = Some(axis);
                face_negative = true;
            }
        } else if empty_cell[axis] >= hit_max[axis] {
            if face_axis.is_none() {
                face_axis = Some(axis);
                face_negative = false;
            }
        }
    }

    let face_axis = face_axis?;

    // Compute placement origin:
    // Non-face axes: snap empty_cell to placement grid via div_euclid
    // Face axis: place adjacent to the hit face
    let mut place_origin = [0i32; 4];
    for axis in 0..4 {
        if axis == face_axis {
            if face_negative {
                place_origin[axis] = hit_min[axis] - place_size;
            } else {
                place_origin[axis] = hit_max[axis];
            }
        } else {
            place_origin[axis] = empty_cell[axis].div_euclid(place_size) * place_size;
        }
    }

    Some(ScaleAwareBlockTarget {
        origin: place_origin,
        scale_exp: placement_scale_exp,
    })
}
