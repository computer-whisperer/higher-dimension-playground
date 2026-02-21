use super::*;

impl Scene {
    fn build_render_region_tree_in_bounds(&self, bounds: Aabb4i) -> RegionChunkTree {
        let mut composed = RegionChunkTree::new();
        if !bounds.is_valid() {
            return composed;
        }
        let world_core = self.world_tree.slice_non_empty_core_in_bounds(bounds);
        let _ = composed.splice_non_empty_core_in_bounds(bounds, &world_core);
        composed
    }

    fn build_render_tree_core_in_bounds(&self, bounds: Aabb4i) -> RenderTreeCore {
        if !bounds.is_valid() {
            return RenderTreeCore::empty(bounds);
        }
        let composed = self.build_render_region_tree_in_bounds(bounds);
        let composed_core = composed.slice_non_empty_core_in_bounds(bounds);
        render_tree::from_region_core(&composed_core)
    }

    fn rebuild_render_bvh_from_region_cache(&mut self, bounds: Aabb4i) {
        let render_core = if let Some(cache) = self.render_region_cache.as_ref() {
            let core = cache.slice_non_empty_core_in_bounds(bounds);
            render_tree::from_region_core(&core)
        } else {
            self.build_render_tree_core_in_bounds(bounds)
        };
        self.render_bvh_cache = Some(render_tree::build_bvh_in_bounds(&render_core, bounds));
        self.render_bvh_cache_bounds = Some(bounds);
    }

    fn intersect_bounds(a: Aabb4i, b: Aabb4i) -> Option<Aabb4i> {
        if !a.is_valid() || !b.is_valid() || !a.intersects(&b) {
            return None;
        }
        Some(Aabb4i::new(
            [
                a.min[0].max(b.min[0]),
                a.min[1].max(b.min[1]),
                a.min[2].max(b.min[2]),
                a.min[3].max(b.min[3]),
            ],
            [
                a.max[0].min(b.max[0]),
                a.max[1].min(b.max[1]),
                a.max[2].min(b.max[2]),
                a.max[3].min(b.max[3]),
            ],
        ))
    }

    fn merge_bounds(a: Aabb4i, b: Aabb4i) -> Aabb4i {
        Aabb4i::new(
            [
                a.min[0].min(b.min[0]),
                a.min[1].min(b.min[1]),
                a.min[2].min(b.min[2]),
                a.min[3].min(b.min[3]),
            ],
            [
                a.max[0].max(b.max[0]),
                a.max[1].max(b.max[1]),
                a.max[2].max(b.max[2]),
                a.max[3].max(b.max[3]),
            ],
        )
    }

    pub(crate) fn bounds_contains_bounds(outer: Aabb4i, inner: Aabb4i) -> bool {
        outer.is_valid()
            && inner.is_valid()
            && outer.min[0] <= inner.min[0]
            && outer.min[1] <= inner.min[1]
            && outer.min[2] <= inner.min[2]
            && outer.min[3] <= inner.min[3]
            && outer.max[0] >= inner.max[0]
            && outer.max[1] >= inner.max[1]
            && outer.max[2] >= inner.max[2]
            && outer.max[3] >= inner.max[3]
    }

    pub(crate) fn ensure_render_bvh_cache_for_bounds(&mut self, bounds: Aabb4i) {
        if !bounds.is_valid() {
            self.render_region_cache_bounds = None;
            self.render_region_cache = None;
            self.render_bvh_cache_bounds = None;
            self.render_bvh_cache = None;
            self.voxel_pending_render_bvh_rebuild = true;
            self.voxel_pending_render_bvh_mutation_deltas.clear();
            return;
        }

        let has_cache = self.render_bvh_cache_bounds == Some(bounds)
            && self.render_region_cache_bounds == Some(bounds)
            && self.render_region_cache.is_some();
        if !has_cache {
            self.render_region_cache_bounds = Some(bounds);
            self.render_region_cache = Some(self.build_render_region_tree_in_bounds(bounds));
            self.rebuild_render_bvh_from_region_cache(bounds);
            self.voxel_pending_render_bvh_rebuild = true;
            self.voxel_pending_render_bvh_mutation_deltas.clear();
            self.clear_voxel_scene_dirty_regions_in_bounds(bounds);
            return;
        }
        if self.voxel_frame_data.region_bvh_root_index == VTE_REGION_BVH_INVALID_NODE
            && self
                .render_bvh_cache
                .as_ref()
                .and_then(|bvh| bvh.root)
                .is_some()
        {
            // Recover from an invalid/empty frame upload by replaying a full
            // snapshot from the already cached render BVH.
            self.voxel_pending_render_bvh_rebuild = true;
        }

        let dirty_regions = self.take_voxel_scene_dirty_regions_in_bounds(
            bounds,
            VOXEL_SCENE_DIRTY_REGION_UPDATE_BUDGET,
        );
        if dirty_regions.is_empty() {
            return;
        }

        let mut changed = false;
        let mut deltas = Vec::<RenderBvhChunkMutationDelta>::new();
        let mut fallback_full_rebuild = false;
        let mut fallback_reason: Option<String> = None;
        for dirty_region in dirty_regions {
            let Some(dirty_bounds) = Self::intersect_bounds(dirty_region, bounds) else {
                continue;
            };
            let desired_fragment_tree = self.build_render_region_tree_in_bounds(dirty_bounds);
            let desired_fragment_core =
                desired_fragment_tree.slice_non_empty_core_in_bounds(dirty_bounds);
            if let Some(cache) = self.render_region_cache.as_mut() {
                let Some(_) =
                    cache.splice_non_empty_core_in_bounds(dirty_bounds, &desired_fragment_core)
                else {
                    continue;
                };
            } else {
                fallback_full_rebuild = true;
                fallback_reason = Some("missing_render_region_cache".to_string());
                break;
            }

            changed = true;
            if let Some(render_bvh) = self.render_bvh_cache.as_mut() {
                let desired_render_core = render_tree::from_region_core(&desired_fragment_core);
                match render_tree::apply_core_patch_in_bounds_with_delta_in_bvh(
                    render_bvh,
                    dirty_bounds,
                    &desired_render_core,
                ) {
                    Ok(Some(delta)) => {
                        deltas.push(delta);
                    }
                    Ok(None) => {}
                    Err(error) => {
                        fallback_full_rebuild = true;
                        fallback_reason = Some(format!(
                            "apply_core_patch_delta_failed bounds={:?}->{:?} error={}",
                            dirty_bounds.min, dirty_bounds.max, error
                        ));
                        break;
                    }
                }
            } else {
                fallback_full_rebuild = true;
                fallback_reason = Some("missing_render_bvh_cache".to_string());
                break;
            }
        }

        if fallback_full_rebuild {
            eprintln!(
                "[vte-bvh-delta-fallback] reason={} rebuild_bounds={:?}->{:?} pending_deltas={}",
                fallback_reason.as_deref().unwrap_or("unknown"),
                bounds.min,
                bounds.max,
                deltas.len()
            );
            self.rebuild_render_bvh_from_region_cache(bounds);
            self.voxel_pending_render_bvh_rebuild = true;
            self.voxel_pending_render_bvh_mutation_deltas.clear();
            return;
        }
        if changed {
            if deltas.is_empty() {
                // If the render-region cache changed but the BVH delta path produced
                // no writes, the GPU frame can get stuck on stale data.
                eprintln!(
                    "[vte-bvh-delta-fallback] reason=changed_cache_without_deltas rebuild_bounds={:?}->{:?}",
                    bounds.min, bounds.max
                );
                self.rebuild_render_bvh_from_region_cache(bounds);
                self.voxel_pending_render_bvh_rebuild = true;
                self.voxel_pending_render_bvh_mutation_deltas.clear();
                return;
            }
            self.voxel_pending_render_bvh_mutation_deltas.extend(deltas);
        }
    }

    pub(crate) fn mark_voxel_scene_region_dirty(&mut self, bounds: Aabb4i) {
        if !bounds.is_valid() {
            return;
        }
        let mut merged = bounds;
        self.voxel_pending_scene_dirty_regions.retain(|existing| {
            if Self::bounds_contains_bounds(*existing, merged) {
                merged = *existing;
                return true;
            }
            if Self::bounds_contains_bounds(merged, *existing) {
                return false;
            }
            if existing.intersects(&merged) {
                merged = Self::merge_bounds(*existing, merged);
                return false;
            }
            true
        });
        self.voxel_pending_scene_dirty_regions.push(merged);
    }

    pub(crate) fn voxel_scene_bounds_has_pending_dirty_regions(&self, bounds: Aabb4i) -> bool {
        if !bounds.is_valid() {
            return false;
        }
        self.voxel_pending_scene_dirty_regions
            .iter()
            .any(|pending| pending.intersects(&bounds))
    }

    fn clear_voxel_scene_dirty_regions_in_bounds(&mut self, bounds: Aabb4i) {
        if !bounds.is_valid() || self.voxel_pending_scene_dirty_regions.is_empty() {
            return;
        }
        self.voxel_pending_scene_dirty_regions
            .retain(|pending| !pending.intersects(&bounds));
    }

    fn take_voxel_scene_dirty_regions_in_bounds(
        &mut self,
        bounds: Aabb4i,
        max_regions: usize,
    ) -> Vec<Aabb4i> {
        if !bounds.is_valid()
            || max_regions == 0
            || self.voxel_pending_scene_dirty_regions.is_empty()
        {
            return Vec::new();
        }

        let mut taken = Vec::<Aabb4i>::new();
        self.voxel_pending_scene_dirty_regions.retain(|pending| {
            if taken.len() < max_regions && pending.intersects(&bounds) {
                taken.push(*pending);
                false
            } else {
                true
            }
        });
        taken
    }

    pub(crate) fn take_pending_render_bvh_update_flags(
        &mut self,
    ) -> (bool, Vec<RenderBvhChunkMutationDelta>) {
        let needs_rebuild = self.voxel_pending_render_bvh_rebuild;
        self.voxel_pending_render_bvh_rebuild = false;
        let deltas = std::mem::take(&mut self.voxel_pending_render_bvh_mutation_deltas);
        (needs_rebuild, deltas)
    }

    pub fn debug_render_bvh_chunk_payloads_in_bounds(
        &mut self,
        bounds: Aabb4i,
        chunk_key: [i32; 4],
    ) -> Vec<ChunkPayload> {
        if !bounds.is_valid() {
            return Vec::new();
        }
        self.ensure_render_bvh_cache_for_bounds(bounds);
        self.render_bvh_cache
            .as_ref()
            .map(|bvh| render_tree::sample_chunk_payloads_from_bvh(bvh, chunk_key))
            .unwrap_or_default()
    }

    pub fn debug_render_bvh_ray_node_hits(
        &mut self,
        ray_origin_world: [f32; 4],
        ray_direction_world: [f32; 4],
        max_distance_world: f32,
        max_nodes: usize,
    ) -> Vec<render_tree::DebugRayBvhNodeHit> {
        if max_nodes == 0 || !max_distance_world.is_finite() || max_distance_world <= 0.0 {
            return Vec::new();
        }
        let active_distance = max_distance_world.max(VOXEL_NEAR_ACTIVE_DISTANCE);
        let chunk_radius = (active_distance / CHUNK_SIZE as f32).ceil() as i32 + 1;
        let chunk_size = CHUNK_SIZE as i32;
        let cam_chunk = [
            (ray_origin_world[0].floor() as i32).div_euclid(chunk_size),
            (ray_origin_world[1].floor() as i32).div_euclid(chunk_size),
            (ray_origin_world[2].floor() as i32).div_euclid(chunk_size),
            (ray_origin_world[3].floor() as i32).div_euclid(chunk_size),
        ];
        let bounds = Aabb4i::new(
            [
                cam_chunk[0] - chunk_radius,
                cam_chunk[1] - chunk_radius,
                cam_chunk[2] - chunk_radius,
                cam_chunk[3] - chunk_radius,
            ],
            [
                cam_chunk[0] + chunk_radius,
                cam_chunk[1] + chunk_radius,
                cam_chunk[2] + chunk_radius,
                cam_chunk[3] + chunk_radius,
            ],
        );
        self.ensure_render_bvh_cache_for_bounds(bounds);
        self.render_bvh_cache
            .as_ref()
            .map(|bvh| {
                render_tree::collect_ray_intersected_nodes_from_bvh(
                    bvh,
                    ray_origin_world,
                    ray_direction_world,
                    max_distance_world,
                    max_nodes,
                )
            })
            .unwrap_or_default()
    }
}
