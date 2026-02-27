use super::*;

impl Scene {
    pub fn apply_region_patch(
        &mut self,
        bounds: Aabb4i,
        subtree: &RegionTreeCore,
    ) -> RegionPatchStats {
        if !bounds.is_valid() {
            return RegionPatchStats::default();
        }

        let collect_previous_start = Instant::now();
        let previous_core = self.world_tree.slice_non_empty_core_in_bounds(bounds);
        let collect_previous_ms = collect_previous_start.elapsed().as_secs_f64() * 1000.0;
        let collect_desired_start = Instant::now();
        let desired_core = slice_non_empty_region_core_in_bounds(subtree, bounds);
        let collect_desired_ms = collect_desired_start.elapsed().as_secs_f64() * 1000.0;
        if previous_core.kind == desired_core.kind
            && previous_core.bounds == desired_core.bounds
        {
            let previous_non_empty = Self::count_non_empty_chunks_in_core(&previous_core);
            return RegionPatchStats {
                previous_non_empty,
                desired_non_empty: previous_non_empty,
                upserts: 0,
                removals: 0,
                changed_chunks: 0,
                previous_total_chunks: previous_non_empty,
                desired_total_chunks: previous_non_empty,
                invalidated_cached_chunks: 0,
                queued_updates: 0,
                collect_previous_ms,
                splice_ms: 0.0,
                collect_desired_ms,
                diff_ms: 0.0,
            };
        }

        let previous_non_empty = Self::count_non_empty_chunks_in_core(&previous_core);
        let desired_non_empty = Self::count_non_empty_chunks_in_core(&desired_core);
        let previous_total_chunks = previous_non_empty;
        let desired_total_chunks = desired_non_empty;
        let diff_ms = 0.0;
        let single_chunk_bounds = bounds.min == bounds.max;
        let previous_single_chunk_payload = if single_chunk_bounds {
            self.world_tree.chunk_payload(bounds.min)
        } else {
            None
        };

        let splice_start = Instant::now();
        let mut changed_bounds = self
            .world_tree
            .splice_non_empty_core_in_bounds(bounds, &desired_core);
        let splice_ms = splice_start.elapsed().as_secs_f64() * 1000.0;
        if changed_bounds.is_none() && single_chunk_bounds {
            let current_single_chunk_payload = self.world_tree.chunk_payload(bounds.min);
            if previous_single_chunk_payload != current_single_chunk_payload {
                changed_bounds = Some(bounds);
                eprintln!(
                    "[scene-region-repair] detected payload change with empty changed-bounds at chunk {:?} (before={:?} after={:?})",
                    bounds.min, previous_single_chunk_payload, current_single_chunk_payload
                );
            }
        }
        let Some(changed_bounds) = changed_bounds else {
            return RegionPatchStats {
                previous_non_empty,
                desired_non_empty,
                upserts: 0,
                removals: 0,
                changed_chunks: 0,
                previous_total_chunks,
                desired_total_chunks,
                invalidated_cached_chunks: 0,
                queued_updates: 0,
                collect_previous_ms,
                splice_ms,
                collect_desired_ms,
                diff_ms,
            };
        };
        self.world_tree_revision = self.world_tree_revision.wrapping_add(1);

        self.mark_voxel_scene_region_dirty(changed_bounds);
        if self.voxel_frame_data.region_bvh_root_index == VTE_REGION_BVH_INVALID_NODE
            && desired_non_empty > 0
        {
            self.voxel_pending_render_bvh_rebuild = true;
        }

        RegionPatchStats {
            previous_non_empty,
            desired_non_empty,
            upserts: 0,
            removals: 0,
            changed_chunks: 1,
            previous_total_chunks,
            desired_total_chunks,
            invalidated_cached_chunks: 0,
            queued_updates: 0,
            collect_previous_ms,
            splice_ms,
            collect_desired_ms,
            diff_ms,
        }
    }

    pub fn apply_region_patch_fast(
        &mut self,
        bounds: Aabb4i,
        subtree: &RegionTreeCore,
    ) -> RegionPatchStats {
        if !bounds.is_valid() {
            return RegionPatchStats::default();
        }

        let desired_owned;
        let desired_core = if subtree.bounds == bounds {
            subtree
        } else {
            desired_owned = slice_non_empty_region_core_in_bounds(subtree, bounds);
            &desired_owned
        };
        let collect_previous_start = Instant::now();
        let previous_core = self.world_tree.slice_non_empty_core_in_bounds(bounds);
        let collect_previous_ms = collect_previous_start.elapsed().as_secs_f64() * 1000.0;
        if previous_core.kind == desired_core.kind
            && previous_core.bounds == desired_core.bounds
        {
            return RegionPatchStats {
                collect_previous_ms,
                ..RegionPatchStats::default()
            };
        }
        let single_chunk_bounds = bounds.min == bounds.max;
        let previous_single_chunk_payload = if single_chunk_bounds {
            self.world_tree.chunk_payload(bounds.min)
        } else {
            None
        };

        let splice_start = Instant::now();
        let mut changed_bounds = self
            .world_tree
            .splice_non_empty_core_in_bounds(bounds, desired_core);
        let splice_ms = splice_start.elapsed().as_secs_f64() * 1000.0;
        if changed_bounds.is_none() && single_chunk_bounds {
            let current_single_chunk_payload = self.world_tree.chunk_payload(bounds.min);
            if previous_single_chunk_payload != current_single_chunk_payload {
                changed_bounds = Some(bounds);
                eprintln!(
                    "[scene-region-repair] detected payload change with empty changed-bounds at chunk {:?} (before={:?} after={:?})",
                    bounds.min, previous_single_chunk_payload, current_single_chunk_payload
                );
            }
        }
        let Some(changed_bounds) = changed_bounds else {
            return RegionPatchStats {
                collect_previous_ms,
                splice_ms,
                ..RegionPatchStats::default()
            };
        };
        self.world_tree_revision = self.world_tree_revision.wrapping_add(1);

        self.mark_voxel_scene_region_dirty(changed_bounds);
        if self.voxel_frame_data.region_bvh_root_index == VTE_REGION_BVH_INVALID_NODE {
            self.voxel_pending_render_bvh_rebuild = true;
        }
        RegionPatchStats {
            changed_chunks: 1,
            invalidated_cached_chunks: 0,
            queued_updates: 0,
            collect_previous_ms,
            splice_ms,
            ..RegionPatchStats::default()
        }
    }

    fn count_non_empty_chunks_in_core(core: &RegionTreeCore) -> usize {
        fn payload_has_non_empty_block(
            payload: &ChunkPayload,
            block_palette: &[polychora::shared::voxel::BlockData],
        ) -> bool {
            let idx_is_solid = |idx: u16| {
                block_palette
                    .get(idx as usize)
                    .map(|b| !b.is_air())
                    .unwrap_or(false)
            };
            match payload {
                ChunkPayload::Empty => false,
                ChunkPayload::Uniform(idx) => idx_is_solid(*idx),
                ChunkPayload::Dense16 { materials } => materials.iter().any(|idx| idx_is_solid(*idx)),
                ChunkPayload::PalettePacked { .. } => payload
                    .dense_materials()
                    .map(|indices| indices.into_iter().any(|idx| idx_is_solid(idx)))
                    .unwrap_or(true),
            }
        }

        fn recurse(core: &RegionTreeCore) -> usize {
            match &core.kind {
                RegionNodeKind::Empty | RegionNodeKind::ProceduralRef(_) => 0,
                RegionNodeKind::Uniform(block) => {
                    if block.is_air() {
                        0
                    } else {
                        core.bounds.chunk_cell_count_at_scale(0).unwrap_or(0)
                    }
                }
                RegionNodeKind::ChunkArray(chunk_array) => {
                    let Ok(indices) = chunk_array.decode_dense_indices() else {
                        return 0;
                    };
                    let bp = &chunk_array.block_palette;
                    let palette_non_empty: Vec<bool> = chunk_array
                        .chunk_palette
                        .iter()
                        .map(|p| payload_has_non_empty_block(p, bp))
                        .collect();
                    indices
                        .into_iter()
                        .filter(|idx| {
                            palette_non_empty
                                .get(*idx as usize)
                                .copied()
                                .unwrap_or(true)
                        })
                        .count()
                }
                RegionNodeKind::Branch(children) => children.iter().map(recurse).sum(),
            }
        }

        recurse(core)
    }
}
