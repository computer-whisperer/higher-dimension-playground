use super::chunk_encoder::{
    chunk_coord_to_i32, leaf_world_bounds, pack_scale_exp_into_orientation,
};
use super::*;
use polychora::content_registry::MaterialResolver;
use polychora::shared::chunk_payload::ResolvedChunkPayload;
use polychora::shared::render_tree::{RenderBvh, RenderBvhNodeKind, RenderLeafKind};
use polychora::shared::spatial::Aabb4i;
use std::time::Instant;

impl Scene {
    pub(super) fn build_voxel_frame_buffers_from_render_bvh(
        render_bvh: &RenderBvh,
        resolver: &MaterialResolver,
    ) -> Result<VoxelFrameDataBuffers, String> {
        let mut voxel_frame_data = VoxelFrameData {
            metadata_generation: 0,
            mutation_base_generation: None,
            region_bvh_root_index: higher_dimension_playground::render::VTE_REGION_BVH_INVALID_NODE,
            dirty_ranges: VoxelFrameDirtyRanges::default(),
            mutation_batch: None,
            chunk_headers: Vec::new(),
            occupancy_words: Vec::new(),
            material_words: Vec::new(),
            orientation_words: Vec::new(),
            macro_words: Vec::new(),
            region_bvh_nodes: Vec::new(),
            leaf_headers: Vec::with_capacity(render_bvh.leaves.len()),
            leaf_chunk_entries: Vec::new(),
        };
        let mut region_bvh_nodes =
            Vec::<GpuVoxelChunkBvhNode>::with_capacity(render_bvh.nodes.len());
        let mut dense_payload_encoded_cache =
            std::collections::HashMap::<DensePayloadCacheKey, u32>::new();

        // Pre-compute leaf world bounds for BVH node encoding.
        let mut per_leaf_world_bounds =
            Vec::<([f32; 4], [f32; 4])>::with_capacity(render_bvh.leaves.len());

        for leaf in &render_bvh.leaves {
            let scale_exp = match &leaf.kind {
                RenderLeafKind::Uniform(block) => block.scale_exp,
                RenderLeafKind::VoxelChunkArray(ca) => ca.scale_exp,
            };
            per_leaf_world_bounds.push(leaf_world_bounds(&leaf.bounds, scale_exp));

            match &leaf.kind {
                RenderLeafKind::Uniform(block) => {
                    let (lat_min, lat_max) = leaf.bounds.to_chunk_lattice_bounds(scale_exp);
                    let mat = resolver.resolve_block(block.namespace, block.block_type);
                    voxel_frame_data.leaf_headers.push(GpuVoxelLeafHeader {
                        min_chunk_coord: lat_min,
                        max_chunk_coord: lat_max,
                        leaf_kind: higher_dimension_playground::render::VTE_LEAF_KIND_UNIFORM,
                        uniform_material: u32::from(mat),
                        chunk_entry_offset: 0,
                        uniform_orientation: pack_scale_exp_into_orientation(
                            u32::from(block.orientation.0),
                            scale_exp,
                        ),
                    });
                }
                RenderLeafKind::VoxelChunkArray(_) => {
                    let (lat_min, lat_max) = leaf.bounds.to_chunk_lattice_bounds(scale_exp);
                    let entry_offset = voxel_frame_data.leaf_chunk_entries.len() as u32;
                    let entries = Self::encode_leaf_chunk_entries(
                        &mut voxel_frame_data,
                        &mut dense_payload_encoded_cache,
                        leaf,
                        resolver,
                    )?;
                    voxel_frame_data
                        .leaf_chunk_entries
                        .extend(entries.into_iter());
                    voxel_frame_data.leaf_headers.push(GpuVoxelLeafHeader {
                        min_chunk_coord: lat_min,
                        max_chunk_coord: lat_max,
                        leaf_kind:
                            higher_dimension_playground::render::VTE_LEAF_KIND_VOXEL_CHUNK_ARRAY,
                        uniform_material: 0,
                        chunk_entry_offset: entry_offset,
                        uniform_orientation: pack_scale_exp_into_orientation(0, scale_exp),
                    });
                }
            }
        }

        // Encode BVH nodes with float world bounds.
        // Children always have lower indices than parents (bottom-up build),
        // so a forward pass gives correct world bounds for internal nodes.
        for node in render_bvh.nodes.iter() {
            let (world_min, world_max) = match node.kind {
                RenderBvhNodeKind::Leaf { leaf_index } => per_leaf_world_bounds
                    .get(leaf_index as usize)
                    .copied()
                    .unwrap_or(leaf_world_bounds(&node.bounds, 0)),
                RenderBvhNodeKind::Internal { left, right } => {
                    let left_bounds = region_bvh_nodes
                        .get(left as usize)
                        .map(|n| (n.world_min, n.world_max));
                    let right_bounds = region_bvh_nodes
                        .get(right as usize)
                        .map(|n| (n.world_min, n.world_max));
                    match (left_bounds, right_bounds) {
                        (Some((lmin, lmax)), Some((rmin, rmax))) => (
                            [
                                lmin[0].min(rmin[0]),
                                lmin[1].min(rmin[1]),
                                lmin[2].min(rmin[2]),
                                lmin[3].min(rmin[3]),
                            ],
                            [
                                lmax[0].max(rmax[0]),
                                lmax[1].max(rmax[1]),
                                lmax[2].max(rmax[2]),
                                lmax[3].max(rmax[3]),
                            ],
                        ),
                        (Some(b), None) | (None, Some(b)) => b,
                        (None, None) => ([0.0; 4], [0.0; 4]),
                    }
                }
            };
            region_bvh_nodes.push(Self::encode_bvh_node(node, world_min, world_max));
        }

        Ok(VoxelFrameDataBuffers {
            region_bvh_root_index: render_bvh
                .root
                .unwrap_or(higher_dimension_playground::render::VTE_REGION_BVH_INVALID_NODE),
            dense_payload_encoded_cache,
            chunk_headers: voxel_frame_data.chunk_headers,
            occupancy_words: voxel_frame_data.occupancy_words,
            material_words: voxel_frame_data.material_words,
            orientation_words: voxel_frame_data.orientation_words,
            macro_words: voxel_frame_data.macro_words,
            region_bvh_nodes,
            leaf_headers: voxel_frame_data.leaf_headers,
            leaf_chunk_entries: voxel_frame_data.leaf_chunk_entries,
        })
    }

    fn clear_voxel_frame_buffers(&mut self) {
        self.active_config.frame_data.region_bvh_root_index =
            higher_dimension_playground::render::VTE_REGION_BVH_INVALID_NODE;
        self.active_config.frame_data.mutation_base_generation = None;
        self.active_config.frame_data.dirty_ranges = VoxelFrameDirtyRanges::default();
        self.active_config.frame_data.mutation_batch = None;
        self.active_config.dense_payload_encoded_cache.clear();
        self.active_config.leaf_entry_spans.clear();
        self.active_config.leaf_entry_free_spans.clear();
        self.active_config.frame_data.chunk_headers.clear();
        self.active_config.frame_data.occupancy_words.clear();
        self.active_config.frame_data.material_words.clear();
        self.active_config.frame_data.orientation_words.clear();
        self.active_config.frame_data.macro_words.clear();
        self.active_config.frame_data.region_bvh_nodes.clear();
        self.active_config.frame_data.leaf_headers.clear();
        self.active_config.frame_data.leaf_chunk_entries.clear();
    }

    fn apply_voxel_frame_buffers(
        &mut self,
        bounds: Aabb4i,
        buffers: Option<VoxelFrameDataBuffers>,
    ) {
        self.voxel_visibility_generation = self.voxel_visibility_generation.wrapping_add(1);
        self.active_config.frame_data.metadata_generation = self.voxel_visibility_generation;
        self.active_config.frame_data.mutation_base_generation = None;
        if let Some(buffers) = buffers {
            self.active_config.frame_data.region_bvh_root_index = buffers.region_bvh_root_index;
            self.active_config.dense_payload_encoded_cache = buffers.dense_payload_encoded_cache;
            self.active_config.frame_data.chunk_headers = buffers.chunk_headers;
            self.active_config.frame_data.occupancy_words = buffers.occupancy_words;
            self.active_config.frame_data.material_words = buffers.material_words;
            self.active_config.frame_data.orientation_words = buffers.orientation_words;
            self.active_config.frame_data.macro_words = buffers.macro_words;
            self.active_config.frame_data.region_bvh_nodes = buffers.region_bvh_nodes;
            self.active_config.frame_data.leaf_headers = buffers.leaf_headers;
            self.active_config.frame_data.leaf_chunk_entries = buffers.leaf_chunk_entries;
            self.active_config.frame_data.mutation_batch = None;
            self.active_config.frame_data.dirty_ranges =
                Self::full_dirty_ranges_from_frame(&self.active_config.frame_data);
            self.sync_leaf_entry_allocator_from_frame();
        } else {
            self.clear_voxel_frame_buffers();
        }
        self.voxel_cached_visibility_bounds = Some(bounds);
    }

    /// Apply voxel frame buffers when the background thread has already created GPU buffers.
    /// Unlike `apply_voxel_frame_buffers`, this does NOT mark dirty ranges — the GPU
    /// buffers are already populated and will be installed into the renderer directly.
    fn apply_voxel_frame_buffers_with_gpu(
        &mut self,
        bounds: Aabb4i,
        buffers: VoxelFrameDataBuffers,
    ) {
        self.voxel_visibility_generation = self.voxel_visibility_generation.wrapping_add(1);
        self.active_config.frame_data.metadata_generation = self.voxel_visibility_generation;
        self.active_config.frame_data.mutation_base_generation = None;
        self.active_config.frame_data.region_bvh_root_index = buffers.region_bvh_root_index;
        self.active_config.dense_payload_encoded_cache = buffers.dense_payload_encoded_cache;
        self.active_config.frame_data.chunk_headers = buffers.chunk_headers;
        self.active_config.frame_data.occupancy_words = buffers.occupancy_words;
        self.active_config.frame_data.material_words = buffers.material_words;
        self.active_config.frame_data.orientation_words = buffers.orientation_words;
        self.active_config.frame_data.macro_words = buffers.macro_words;
        self.active_config.frame_data.region_bvh_nodes = buffers.region_bvh_nodes;
        self.active_config.frame_data.leaf_headers = buffers.leaf_headers;
        self.active_config.frame_data.leaf_chunk_entries = buffers.leaf_chunk_entries;
        self.active_config.frame_data.mutation_batch = None;
        // No dirty ranges — GPU buffers are pre-populated.
        self.active_config.frame_data.dirty_ranges = VoxelFrameDirtyRanges::default();
        self.sync_leaf_entry_allocator_from_frame();
        self.voxel_cached_visibility_bounds = Some(bounds);
    }

    /// Build the voxel-native frame payload for VTE.
    pub fn build_voxel_frame_data(
        &mut self,
        cam_pos: [f32; 4],
        _cam_forward: [f32; 4],
        max_trace_distance: f32,
        resolver: &MaterialResolver,
    ) -> VoxelFrameBuildResult<'_> {
        const SCENE_RESIDENCY_RADIUS_MULTIPLIER: i32 = 2;
        const SCENE_RESIDENCY_EXTRA_CHUNKS: i32 = 2;
        let mut swap_gpu_buffers: Option<higher_dimension_playground::render::VoxelGpuBuffers> =
            None;
        let mut swap_gpu_generation: Option<u64> = None;

        let active_distance = max_trace_distance.max(VOXEL_NEAR_ACTIVE_DISTANCE);
        let chunk_radius = (active_distance / CHUNK_SIZE as f32).ceil() as i32 + 1;
        let cam_chunk = Self::camera_chunk_key(cam_pos);
        let view_bounds = Aabb4i::from_lattice_bounds(
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
            0,
        );
        let resident_radius = chunk_radius
            .saturating_mul(SCENE_RESIDENCY_RADIUS_MULTIPLIER)
            .saturating_add(SCENE_RESIDENCY_EXTRA_CHUNKS)
            .max(chunk_radius + 1);
        let desired_scene_bounds = Aabb4i::from_lattice_bounds(
            [
                cam_chunk[0] - resident_radius,
                cam_chunk[1] - resident_radius,
                cam_chunk[2] - resident_radius,
                cam_chunk[3] - resident_radius,
            ],
            [
                cam_chunk[0] + resident_radius,
                cam_chunk[1] + resident_radius,
                cam_chunk[2] + resident_radius,
                cam_chunk[3] + resident_radius,
            ],
            0,
        );
        // Poll background rebuild before computing scene_bounds, since applying
        // the result updates voxel_cached_visibility_bounds.
        if let Some(bg) = self.voxel_background_rebuild.take() {
            match bg.receiver.try_recv() {
                Ok(Ok(result)) => {
                    eprintln!(
                        "[voxel-bg-rebuild] background build completed, applying for bounds {:?}->{:?}",
                        bg.bounds.min, bg.bounds.max
                    );
                    self.active_config.render_bvh_cache = Some(result.render_bvh);
                    self.active_config.render_bvh_cache_bounds = Some(bg.bounds);
                    if result.gpu_buffers.is_some() {
                        // Fast path: GPU buffers pre-built on background thread.
                        self.apply_voxel_frame_buffers_with_gpu(bg.bounds, result.frame_buffers);
                        swap_gpu_buffers = result.gpu_buffers;
                        // Capture the generation at swap time. If deltas are applied
                        // later in this same call, frame_data.metadata_generation will
                        // advance, but the GPU buffers still contain this generation's
                        // data. The caller must use this value so the renderer correctly
                        // detects the delta as dirty and uploads it.
                        swap_gpu_generation =
                            Some(self.active_config.frame_data.metadata_generation);
                    } else {
                        // Fallback: no allocator available (e.g. preload before renderer init).
                        // Use the old dirty-range path — renderer will upload from CPU data.
                        self.apply_voxel_frame_buffers(bg.bounds, Some(result.frame_buffers));
                    }
                    self.active_config.pending_render_bvh_rebuild = false;
                    // Fall through to normal processing for any dirty regions accumulated during build
                }
                Ok(Err(error)) => {
                    eprintln!("[voxel-bg-rebuild] background build failed: {error}");
                    self.active_config.pending_render_bvh_rebuild = true;
                }
                Err(std::sync::mpsc::TryRecvError::Disconnected) => {
                    eprintln!("[voxel-bg-rebuild] background thread panicked");
                    self.active_config.pending_render_bvh_rebuild = true;
                }
                Err(std::sync::mpsc::TryRecvError::Empty) => {
                    // Still building — return stale frame data, skip all processing
                    self.voxel_background_rebuild = Some(bg);
                    return VoxelFrameBuildResult {
                        frame_data: &self.active_config.frame_data,
                        new_gpu_buffers: None,
                        gpu_buffers_generation: None,
                    };
                }
            }
        }

        // Compute scene_bounds AFTER polling the background rebuild, so that
        // voxel_cached_visibility_bounds reflects any just-applied result.
        let scene_bounds = self
            .voxel_cached_visibility_bounds
            .filter(|active_bounds| active_bounds.contains_bounds(&view_bounds))
            .unwrap_or(desired_scene_bounds);

        let frame_root_valid = self.voxel_frame_root_is_valid();
        let cached_render_bvh_empty = self
            .active_config
            .render_bvh_cache
            .as_ref()
            .map(|bvh| bvh.root.is_none())
            .unwrap_or(false);
        let has_pending_render_updates = self.active_config.pending_render_bvh_rebuild
            || !self
                .active_config
                .pending_render_bvh_mutation_deltas
                .is_empty();
        let visibility_cache_valid = self.voxel_cached_visibility_bounds == Some(scene_bounds)
            && !self.voxel_scene_bounds_has_pending_dirty_regions(scene_bounds)
            && !has_pending_render_updates
            && (frame_root_valid || cached_render_bvh_empty);
        if std::env::var_os("R4D_EDIT_RENDER_SYNC_DIAG").is_some()
            && !self.voxel_pending_scene_dirty_regions.is_empty()
        {
            let pending_intersects =
                self.voxel_scene_bounds_has_pending_dirty_regions(scene_bounds);
            if visibility_cache_valid || !pending_intersects {
                eprintln!(
                    "[edit-sync-voxel-build] scene_bounds={:?}->{:?} cached_bounds={:?} pending_total={} pending_intersects={} visibility_cache_valid={} frame_root_valid={} cached_bvh_empty={}",
                    scene_bounds.min,
                    scene_bounds.max,
                    self.voxel_cached_visibility_bounds,
                    self.voxel_pending_scene_dirty_regions.len(),
                    pending_intersects,
                    visibility_cache_valid,
                    frame_root_valid,
                    cached_render_bvh_empty
                );
            }
        }
        if std::env::var_os("R4D_EDIT_RENDER_SYNC_DIAG").is_some()
            && has_pending_render_updates
            && self.voxel_cached_visibility_bounds == Some(scene_bounds)
            && !self.voxel_scene_bounds_has_pending_dirty_regions(scene_bounds)
        {
            eprintln!(
                "[edit-sync-voxel-build] scene_bounds={:?}->{:?} forcing_update_for_pending_render_changes pending_rebuild={} pending_deltas={}",
                scene_bounds.min,
                scene_bounds.max,
                self.active_config.pending_render_bvh_rebuild,
                self.active_config.pending_render_bvh_mutation_deltas.len()
            );
        }
        if !visibility_cache_valid {
            self.ensure_render_bvh_cache_for_bounds(scene_bounds);
            let (needs_rebuild, deltas) = self.take_pending_render_bvh_update_flags();
            if std::env::var_os("R4D_EDIT_RENDER_SYNC_DIAG").is_some() {
                eprintln!(
                    "[edit-sync-voxel-flags] scene_bounds={:?}->{:?} needs_rebuild={} deltas={} frame_root={} cache_root={:?}",
                    scene_bounds.min,
                    scene_bounds.max,
                    needs_rebuild,
                    deltas.len(),
                    self.active_config.frame_data.region_bvh_root_index,
                    self.active_config.render_bvh_cache.as_ref().and_then(|bvh| bvh.root),
                );
            }

            if needs_rebuild {
                self.log_voxel_snapshot_rebuild(
                    scene_bounds,
                    "pending_rebuild_flag",
                    0,
                    self.active_config.pending_render_bvh_mutation_deltas.len(),
                    self.active_config.frame_data.region_bvh_root_index,
                    self.active_config
                        .render_bvh_cache
                        .as_ref()
                        .and_then(|bvh| bvh.root)
                        .unwrap_or(
                            higher_dimension_playground::render::VTE_REGION_BVH_INVALID_NODE,
                        ),
                );
                // Clone the world tree slice and spawn the full chain on a background thread:
                // world_tree slice → RenderTreeCore → RenderBvh → GPU encode → GPU buffers
                let world_core = self.world_tree.slice_non_empty_core_in_bounds(scene_bounds);
                let resolver_clone = resolver.clone();
                let bg_bounds = scene_bounds;
                let bg_allocator = self.memory_allocator.clone();
                let (tx, rx) = std::sync::mpsc::channel();
                std::thread::spawn(move || {
                    let result = (|| -> Result<BackgroundRebuildResult, String> {
                        let render_core = render_tree::from_region_core(&world_core);
                        let render_bvh = render_tree::build_bvh_in_bounds(&render_core, bg_bounds);
                        let frame_buffers = Scene::build_voxel_frame_buffers_from_render_bvh(
                            &render_bvh,
                            &resolver_clone,
                        )?;
                        // Create pre-populated GPU buffers on the background thread
                        // to avoid blocking the main thread with synchronous uploads.
                        let gpu_buffers = bg_allocator.map(|allocator| {
                            use higher_dimension_playground::render::VoxelGpuBuffers;
                            VoxelGpuBuffers::from_data(
                                allocator,
                                &frame_buffers.chunk_headers,
                                &frame_buffers.occupancy_words,
                                &frame_buffers.material_words,
                                &frame_buffers.orientation_words,
                                &frame_buffers.macro_words,
                                &frame_buffers.leaf_headers,
                                &frame_buffers.region_bvh_nodes,
                                &frame_buffers.leaf_chunk_entries,
                            )
                        });
                        Ok(BackgroundRebuildResult {
                            render_bvh,
                            frame_buffers,
                            gpu_buffers,
                        })
                    })();
                    let _ = tx.send(result);
                });
                self.voxel_background_rebuild = Some(BackgroundVoxelRebuild {
                    receiver: rx,
                    bounds: scene_bounds,
                });
            } else if !deltas.is_empty() {
                let applied_deltas = deltas.len();
                let mut apply_ok = false;
                let mut apply_error: Option<String> = None;
                if let Some(render_bvh) = self.active_config.render_bvh_cache.take() {
                    match self.apply_render_bvh_mutation_deltas_to_voxel_frame_data(
                        &deltas,
                        scene_bounds,
                        resolver,
                    ) {
                        Ok(()) => {
                            apply_ok = true;
                        }
                        Err(error) => {
                            apply_error = Some(error);
                        }
                    }
                    self.active_config.render_bvh_cache = Some(render_bvh);
                }

                if !apply_ok {
                    if std::env::var_os("R4D_EDIT_RENDER_SYNC_DIAG").is_some() {
                        eprintln!(
                            "[edit-sync-voxel-delta] scene_bounds={:?}->{:?} result=error error={:?}",
                            scene_bounds.min,
                            scene_bounds.max,
                            apply_error
                        );
                    }
                    self.active_config
                        .pending_render_bvh_mutation_deltas
                        .clear();
                    // Schedule a background rebuild instead of synchronous recovery
                    self.active_config.pending_render_bvh_rebuild = true;
                    eprintln!(
                        "[vte-delta-fallback] scheduling background rebuild reason={:?} applied_deltas={}",
                        apply_error, applied_deltas
                    );
                } else if std::env::var_os("R4D_EDIT_RENDER_SYNC_DIAG").is_some() {
                    eprintln!(
                        "[edit-sync-voxel-delta] scene_bounds={:?}->{:?} result=ok applied_deltas={} frame_root={}",
                        scene_bounds.min,
                        scene_bounds.max,
                        applied_deltas,
                        self.active_config.frame_data.region_bvh_root_index
                    );
                    for delta in deltas.iter().take(8) {
                        let key = delta.key;
                        let world_payload = self.world_tree.chunk_payload(key).map(|(p, _)| p);
                        let bvh_payloads = self
                            .active_config
                            .render_bvh_cache
                            .as_ref()
                            .map(|bvh| render_tree::sample_chunk_payloads_from_bvh(bvh, key))
                            .unwrap_or_default();
                        let frame_payloads: Vec<ResolvedChunkPayload> = self
                            .debug_voxel_frame_chunk_payloads(chunk_coord_to_i32(key))
                            .into_iter()
                            .map(ResolvedChunkPayload::from_payload_with_static_palette)
                            .collect();
                        eprintln!(
                            "[edit-sync-render] key={:?} root={:?}->{:?} writes(nodes={},leaves={},freed_nodes={},freed_leaves={}) world={} bvh={} frame={}",
                            key,
                            delta.expected_root,
                            delta.new_root,
                            delta.node_writes.len(),
                            delta.leaf_writes.len(),
                            delta.freed_node_ids.len(),
                            delta.freed_leaf_ids.len(),
                            Self::summarize_chunk_payload_compact(world_payload.as_ref()),
                            Self::summarize_chunk_payload_list_compact(&bvh_payloads),
                            Self::summarize_chunk_payload_list_compact(&frame_payloads),
                        );
                    }
                }
            }
        }

        VoxelFrameBuildResult {
            frame_data: &self.active_config.frame_data,
            new_gpu_buffers: swap_gpu_buffers,
            gpu_buffers_generation: swap_gpu_generation,
        }
    }

    /// Block until any in-flight background voxel rebuild completes and apply the result.
    pub(crate) fn flush_voxel_background_rebuild(&mut self) {
        if let Some(bg) = self.voxel_background_rebuild.take() {
            match bg.receiver.recv() {
                Ok(Ok(result)) => {
                    self.active_config.render_bvh_cache = Some(result.render_bvh);
                    self.active_config.render_bvh_cache_bounds = Some(bg.bounds);
                    self.apply_voxel_frame_buffers(bg.bounds, Some(result.frame_buffers));
                    self.active_config.pending_render_bvh_rebuild = false;
                }
                Ok(Err(error)) => {
                    eprintln!("[voxel-bg-rebuild] background build failed: {error}");
                    self.active_config.pending_render_bvh_rebuild = true;
                }
                Err(_) => {
                    eprintln!("[voxel-bg-rebuild] background thread panicked");
                    self.active_config.pending_render_bvh_rebuild = true;
                }
            }
        }
    }

    /// Prime voxel frame metadata around the current spawn/camera position.
    pub fn preload_spawn_chunks(
        &mut self,
        spawn_pos: [f32; 4],
        max_trace_distance: f32,
        resolver: &MaterialResolver,
    ) {
        let start = Instant::now();
        let _ = self.build_voxel_frame_data(
            spawn_pos,
            [0.0, 0.0, 1.0, 0.0],
            max_trace_distance,
            resolver,
        );
        self.flush_voxel_background_rebuild();
        eprintln!(
            "Preloaded render-tree voxel metadata in {:.2} ms",
            start.elapsed().as_secs_f64() * 1000.0
        );
    }
}
