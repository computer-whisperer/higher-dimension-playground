use super::chunk_encoder::{leaf_world_bounds, pack_scale_exp_into_orientation};
use super::*;
use polychora::content_registry::MaterialResolver;
use polychora::shared::render_tree::{RenderBvhChunkMutationDelta, RenderLeafKind};
use polychora::shared::spatial::Aabb4i;

impl Scene {
    pub(super) fn invalid_leaf_header() -> GpuVoxelLeafHeader {
        GpuVoxelLeafHeader {
            min_chunk_coord: [0, 0, 0, 0],
            max_chunk_coord: [-1, -1, -1, -1],
            leaf_kind: higher_dimension_playground::render::VTE_LEAF_KIND_UNIFORM,
            uniform_material: 0,
            chunk_entry_offset: 0,
            uniform_orientation: 0,
        }
    }

    pub(super) fn mark_dirty_range(
        slot: &mut Option<std::ops::Range<usize>>,
        range: std::ops::Range<usize>,
    ) {
        if range.start >= range.end {
            return;
        }
        match slot {
            Some(existing) => {
                existing.start = existing.start.min(range.start);
                existing.end = existing.end.max(range.end);
            }
            None => *slot = Some(range),
        }
    }

    pub(super) fn mark_dirty_index(slot: &mut Option<std::ops::Range<usize>>, index: usize) {
        Self::mark_dirty_range(slot, index..index.saturating_add(1));
    }

    pub(super) fn mark_appended_tail(
        slot: &mut Option<std::ops::Range<usize>>,
        old_len: usize,
        new_len: usize,
    ) {
        if new_len > old_len {
            Self::mark_dirty_range(slot, old_len..new_len);
        }
    }

    pub(super) fn full_dirty_ranges_from_frame(
        voxel_frame_data: &VoxelFrameData,
    ) -> VoxelFrameDirtyRanges {
        let mut dirty = VoxelFrameDirtyRanges::default();
        if !voxel_frame_data.chunk_headers.is_empty() {
            dirty.chunk_headers = Some(0..voxel_frame_data.chunk_headers.len());
        }
        if !voxel_frame_data.occupancy_words.is_empty() {
            dirty.occupancy_words = Some(0..voxel_frame_data.occupancy_words.len());
        }
        if !voxel_frame_data.material_words.is_empty() {
            dirty.material_words = Some(0..voxel_frame_data.material_words.len());
        }
        if !voxel_frame_data.orientation_words.is_empty() {
            dirty.orientation_words = Some(0..voxel_frame_data.orientation_words.len());
        }
        if !voxel_frame_data.macro_words.is_empty() {
            dirty.macro_words = Some(0..voxel_frame_data.macro_words.len());
        }
        if !voxel_frame_data.region_bvh_nodes.is_empty() {
            dirty.region_bvh_nodes = Some(0..voxel_frame_data.region_bvh_nodes.len());
        }
        if !voxel_frame_data.leaf_headers.is_empty() {
            dirty.leaf_headers = Some(0..voxel_frame_data.leaf_headers.len());
        }
        if !voxel_frame_data.leaf_chunk_entries.is_empty() {
            dirty.leaf_chunk_entries = Some(0..voxel_frame_data.leaf_chunk_entries.len());
        }
        dirty
    }

    fn chunk_header_range_write_from_dirty(
        chunk_headers: &[GpuVoxelChunkHeader],
        dirty: Option<std::ops::Range<usize>>,
    ) -> Option<higher_dimension_playground::render::VoxelChunkHeaderRangeWrite> {
        let range = dirty?;
        if range.start >= range.end || range.end > chunk_headers.len() {
            return None;
        }
        Some(
            higher_dimension_playground::render::VoxelChunkHeaderRangeWrite {
                start: range.start as u32,
                values: chunk_headers[range].to_vec(),
            },
        )
    }

    fn bvh_node_range_write_from_dirty(
        nodes: &[GpuVoxelChunkBvhNode],
        dirty: Option<std::ops::Range<usize>>,
    ) -> Option<higher_dimension_playground::render::VoxelChunkBvhNodeRangeWrite> {
        let range = dirty?;
        if range.start >= range.end || range.end > nodes.len() {
            return None;
        }
        Some(
            higher_dimension_playground::render::VoxelChunkBvhNodeRangeWrite {
                start: range.start as u32,
                values: nodes[range].to_vec(),
            },
        )
    }

    fn leaf_header_range_write_from_dirty(
        leaf_headers: &[GpuVoxelLeafHeader],
        dirty: Option<std::ops::Range<usize>>,
    ) -> Option<higher_dimension_playground::render::VoxelLeafHeaderRangeWrite> {
        let range = dirty?;
        if range.start >= range.end || range.end > leaf_headers.len() {
            return None;
        }
        Some(
            higher_dimension_playground::render::VoxelLeafHeaderRangeWrite {
                start: range.start as u32,
                values: leaf_headers[range].to_vec(),
            },
        )
    }

    fn u32_range_write_from_dirty(
        values: &[u32],
        dirty: Option<std::ops::Range<usize>>,
    ) -> Option<higher_dimension_playground::render::VoxelU32RangeWrite> {
        let range = dirty?;
        if range.start >= range.end || range.end > values.len() {
            return None;
        }
        Some(higher_dimension_playground::render::VoxelU32RangeWrite {
            start: range.start as u32,
            values: values[range].to_vec(),
        })
    }

    pub(super) fn build_mutation_batch_from_dirty_ranges(
        voxel_frame_data: &VoxelFrameData,
        dirty: &VoxelFrameDirtyRanges,
    ) -> Option<higher_dimension_playground::render::VoxelMutationBatch> {
        let mut batch = higher_dimension_playground::render::VoxelMutationBatch::default();
        if let Some(write) = Self::chunk_header_range_write_from_dirty(
            &voxel_frame_data.chunk_headers,
            dirty.chunk_headers.clone(),
        ) {
            batch.chunk_header_writes.push(write);
        }
        if let Some(write) = Self::u32_range_write_from_dirty(
            &voxel_frame_data.occupancy_words,
            dirty.occupancy_words.clone(),
        ) {
            batch.occupancy_word_writes.push(write);
        }
        if let Some(write) = Self::u32_range_write_from_dirty(
            &voxel_frame_data.material_words,
            dirty.material_words.clone(),
        ) {
            batch.material_word_writes.push(write);
        }
        if let Some(write) = Self::u32_range_write_from_dirty(
            &voxel_frame_data.orientation_words,
            dirty.orientation_words.clone(),
        ) {
            batch.orientation_word_writes.push(write);
        }
        if let Some(write) = Self::u32_range_write_from_dirty(
            &voxel_frame_data.macro_words,
            dirty.macro_words.clone(),
        ) {
            batch.macro_word_writes.push(write);
        }
        if let Some(write) = Self::bvh_node_range_write_from_dirty(
            &voxel_frame_data.region_bvh_nodes,
            dirty.region_bvh_nodes.clone(),
        ) {
            batch.region_bvh_node_writes.push(write);
        }
        if let Some(write) = Self::leaf_header_range_write_from_dirty(
            &voxel_frame_data.leaf_headers,
            dirty.leaf_headers.clone(),
        ) {
            batch.leaf_header_writes.push(write);
        }
        if let Some(write) = Self::u32_range_write_from_dirty(
            &voxel_frame_data.leaf_chunk_entries,
            dirty.leaf_chunk_entries.clone(),
        ) {
            batch.leaf_chunk_entry_writes.push(write);
        }

        let has_ops = !batch.chunk_header_writes.is_empty()
            || !batch.occupancy_word_writes.is_empty()
            || !batch.material_word_writes.is_empty()
            || !batch.orientation_word_writes.is_empty()
            || !batch.macro_word_writes.is_empty()
            || !batch.region_bvh_node_writes.is_empty()
            || !batch.leaf_header_writes.is_empty()
            || !batch.leaf_chunk_entry_writes.is_empty();
        has_ops.then_some(batch)
    }

    pub(super) fn build_leaf_entry_spans_from_headers(
        voxel_frame_data: &VoxelFrameData,
    ) -> Vec<Option<std::ops::Range<usize>>> {
        let mut out = vec![None; voxel_frame_data.leaf_headers.len()];
        for (idx, header) in voxel_frame_data.leaf_headers.iter().enumerate() {
            if header.leaf_kind
                != higher_dimension_playground::render::VTE_LEAF_KIND_VOXEL_CHUNK_ARRAY
            {
                continue;
            }
            // min_chunk_coord / max_chunk_coord are lattice-space integers.
            // Compute cell count directly from lattice dimensions.
            let mut cell_count = 1usize;
            let mut valid = true;
            for axis in 0..4 {
                let span =
                    header.max_chunk_coord[axis] as i64 - header.min_chunk_coord[axis] as i64 + 1;
                if span <= 0 {
                    valid = false;
                    break;
                }
                cell_count = cell_count.saturating_mul(span as usize);
            }
            if !valid || cell_count == 0 {
                continue;
            }
            let start = header.chunk_entry_offset as usize;
            let end = start.saturating_add(cell_count);
            if end > voxel_frame_data.leaf_chunk_entries.len() {
                continue;
            }
            out[idx] = Some(start..end);
        }
        out
    }

    pub(super) fn sync_leaf_entry_allocator_from_frame(&mut self) {
        let spans = Self::build_leaf_entry_spans_from_headers(&self.active_config.frame_data);
        let free_spans = Self::build_free_spans_from_leaf_spans(
            &spans,
            self.active_config.frame_data.leaf_chunk_entries.len(),
        );
        self.active_config.leaf_entry_spans = spans;
        self.active_config.leaf_entry_free_spans = free_spans;
    }

    pub(super) fn merge_free_span(
        free_spans: &mut Vec<std::ops::Range<usize>>,
        new_span: std::ops::Range<usize>,
    ) {
        if new_span.start >= new_span.end {
            return;
        }
        // Insert in sorted position (by start), then merge with neighbors.
        let pos = free_spans
            .binary_search_by_key(&new_span.start, |s| s.start)
            .unwrap_or_else(|p| p);
        free_spans.insert(pos, new_span);

        // Merge with the following span if they overlap/touch.
        while pos + 1 < free_spans.len() && free_spans[pos + 1].start <= free_spans[pos].end {
            free_spans[pos].end = free_spans[pos].end.max(free_spans[pos + 1].end);
            free_spans.remove(pos + 1);
        }
        // Merge with the preceding span if they overlap/touch.
        if pos > 0 && free_spans[pos - 1].end >= free_spans[pos].start {
            free_spans[pos - 1].end = free_spans[pos - 1].end.max(free_spans[pos].end);
            free_spans.remove(pos);
        }
    }

    pub(super) fn build_free_spans_from_leaf_spans(
        spans: &[Option<std::ops::Range<usize>>],
        used_len: usize,
    ) -> Vec<std::ops::Range<usize>> {
        let mut used = spans
            .iter()
            .filter_map(|span| span.clone())
            .collect::<Vec<std::ops::Range<usize>>>();
        used.sort_unstable_by_key(|span| span.start);
        let mut free = Vec::<std::ops::Range<usize>>::new();
        let mut cursor = 0usize;
        for span in used {
            if span.start > cursor {
                free.push(cursor..span.start);
            }
            cursor = cursor.max(span.end);
        }
        if cursor < used_len {
            free.push(cursor..used_len);
        }
        free
    }

    pub(super) fn allocate_leaf_entry_span(
        free_spans: &mut Vec<std::ops::Range<usize>>,
        leaf_chunk_entries: &mut Vec<u32>,
        len: usize,
    ) -> Result<std::ops::Range<usize>, String> {
        if len == 0 {
            return Ok(0..0);
        }
        let mut selected_idx = None;
        for (idx, span) in free_spans.iter().enumerate() {
            if span.end.saturating_sub(span.start) >= len {
                selected_idx = Some(idx);
                break;
            }
        }
        let Some(span_idx) = selected_idx else {
            let start = leaf_chunk_entries.len();
            let end = start.saturating_add(len);
            leaf_chunk_entries.resize(
                end,
                higher_dimension_playground::render::VTE_LEAF_CHUNK_ENTRY_EMPTY,
            );
            return Ok(start..end);
        };
        let span = free_spans.remove(span_idx);
        let alloc = span.start..(span.start + len);
        if alloc.end < span.end {
            free_spans.push(alloc.end..span.end);
        }
        if leaf_chunk_entries.len() < alloc.end {
            leaf_chunk_entries.resize(
                alloc.end,
                higher_dimension_playground::render::VTE_LEAF_CHUNK_ENTRY_EMPTY,
            );
        }
        Ok(alloc)
    }

    pub(super) fn apply_render_bvh_mutation_deltas_to_voxel_frame_data(
        &mut self,
        deltas: &[RenderBvhChunkMutationDelta],
        bounds: Aabb4i,
        resolver: &MaterialResolver,
    ) -> Result<(), String> {
        if !bounds.is_valid() {
            return Ok(());
        }
        if self.active_config.leaf_entry_spans.len()
            != self.active_config.frame_data.leaf_headers.len()
        {
            self.sync_leaf_entry_allocator_from_frame();
        }
        let leaf_spans = &mut self.active_config.leaf_entry_spans;
        let free_spans = &mut self.active_config.leaf_entry_free_spans;
        let old_chunk_headers_len = self.active_config.frame_data.chunk_headers.len();
        let old_occupancy_words_len = self.active_config.frame_data.occupancy_words.len();
        let old_material_words_len = self.active_config.frame_data.material_words.len();
        let old_orientation_words_len = self.active_config.frame_data.orientation_words.len();
        let old_macro_words_len = self.active_config.frame_data.macro_words.len();
        // Start from the previous cumulative dirty_ranges so that frame-in-flight
        // slots that missed earlier deltas can still apply the mutation batch
        // (which is built from these ranges and contains all changes since the
        // last full rebuild).
        let mut dirty = std::mem::take(&mut self.active_config.frame_data.dirty_ranges);
        let mut current_root = if self.active_config.frame_data.region_bvh_root_index
            == higher_dimension_playground::render::VTE_REGION_BVH_INVALID_NODE
        {
            None
        } else {
            Some(self.active_config.frame_data.region_bvh_root_index)
        };

        for (delta_index, delta) in deltas.iter().enumerate() {
            if delta.expected_root != current_root {
                return Err(format!(
                    "delta root mismatch at index {} (expected_root={:?}, frame_root={:?}, new_root={:?})",
                    delta_index,
                    delta.expected_root,
                    current_root,
                    delta.new_root,
                ));
            }
            // Free old resources first.
            for node_index in &delta.freed_node_ids {
                let node_index = *node_index as usize;
                if node_index < self.active_config.frame_data.region_bvh_nodes.len() {
                    self.active_config.frame_data.region_bvh_nodes[node_index] =
                        GpuVoxelChunkBvhNode::empty();
                    Self::mark_dirty_index(&mut dirty.region_bvh_nodes, node_index);
                }
            }

            for leaf_index in &delta.freed_leaf_ids {
                let leaf_index = *leaf_index as usize;
                if leaf_index < leaf_spans.len() {
                    if let Some(old_span) = leaf_spans[leaf_index].take() {
                        Self::merge_free_span(free_spans, old_span);
                    }
                }
                if leaf_index < self.active_config.frame_data.leaf_headers.len() {
                    self.active_config.frame_data.leaf_headers[leaf_index] =
                        Self::invalid_leaf_header();
                    Self::mark_dirty_index(&mut dirty.leaf_headers, leaf_index);
                }
            }

            for (leaf_index, leaf) in &delta.leaf_writes {
                let leaf_index = *leaf_index as usize;
                if self.active_config.frame_data.leaf_headers.len() <= leaf_index {
                    self.active_config
                        .frame_data
                        .leaf_headers
                        .resize(leaf_index + 1, Self::invalid_leaf_header());
                }
                if leaf_spans.len() <= leaf_index {
                    leaf_spans.resize(leaf_index + 1, None);
                }
                if let Some(old_span) = leaf_spans[leaf_index].take() {
                    Self::merge_free_span(free_spans, old_span);
                }

                match &leaf.kind {
                    RenderLeafKind::Uniform(block) => {
                        let scale_exp = block.scale_exp;
                        let (lat_min, lat_max) = leaf.bounds.to_chunk_lattice_bounds(scale_exp);
                        let mat = resolver.resolve_block(block.namespace, block.block_type);
                        self.active_config.frame_data.leaf_headers[leaf_index] =
                            GpuVoxelLeafHeader {
                                min_chunk_coord: lat_min,
                                max_chunk_coord: lat_max,
                                leaf_kind:
                                    higher_dimension_playground::render::VTE_LEAF_KIND_UNIFORM,
                                uniform_material: u32::from(mat),
                                chunk_entry_offset: 0,
                                uniform_orientation: pack_scale_exp_into_orientation(
                                    u32::from(block.orientation.0),
                                    scale_exp,
                                ),
                            };
                        Self::mark_dirty_index(&mut dirty.leaf_headers, leaf_index);
                    }
                    RenderLeafKind::VoxelChunkArray(chunk_array) => {
                        let scale_exp = chunk_array.scale_exp;
                        let (lat_min, lat_max) = leaf.bounds.to_chunk_lattice_bounds(scale_exp);
                        let entries = Self::encode_leaf_chunk_entries(
                            &mut self.active_config.frame_data,
                            &mut self.active_config.dense_payload_encoded_cache,
                            leaf,
                            resolver,
                        )?;
                        let span = Self::allocate_leaf_entry_span(
                            free_spans,
                            &mut self.active_config.frame_data.leaf_chunk_entries,
                            entries.len(),
                        )?;
                        self.active_config.frame_data.leaf_chunk_entries[span.clone()]
                            .copy_from_slice(entries.as_slice());
                        Self::mark_dirty_range(&mut dirty.leaf_chunk_entries, span.clone());
                        leaf_spans[leaf_index] = Some(span.clone());
                        self.active_config.frame_data.leaf_headers[leaf_index] = GpuVoxelLeafHeader {
                            min_chunk_coord: lat_min,
                            max_chunk_coord: lat_max,
                            leaf_kind:
                                higher_dimension_playground::render::VTE_LEAF_KIND_VOXEL_CHUNK_ARRAY,
                            uniform_material: 0,
                            chunk_entry_offset: span.start as u32,
                            uniform_orientation: pack_scale_exp_into_orientation(0, scale_exp),
                        };
                        Self::mark_dirty_index(&mut dirty.leaf_headers, leaf_index);
                    }
                }
            }

            // Collect leaf world bounds for newly-written leaves so BVH nodes
            // can reference them when computing world bounds.
            let mut leaf_world_bounds_map =
                std::collections::HashMap::<u32, ([f32; 4], [f32; 4])>::new();
            for (leaf_index, leaf) in &delta.leaf_writes {
                let scale_exp = match &leaf.kind {
                    RenderLeafKind::Uniform(block) => block.scale_exp,
                    RenderLeafKind::VoxelChunkArray(ca) => ca.scale_exp,
                };
                leaf_world_bounds_map
                    .insert(*leaf_index, leaf_world_bounds(&leaf.bounds, scale_exp));
            }

            // Write BVH nodes after leaves so we can compute world bounds.
            // Pre-compute all node bounds recursively so that internal nodes
            // resolve their children correctly regardless of index order (freed
            // node IDs can be reused in arbitrary order by the allocator).
            let computed_node_bounds = Self::compute_all_delta_node_world_bounds(
                &delta.node_writes,
                &self.active_config.frame_data.region_bvh_nodes,
                &self.active_config.frame_data.leaf_headers,
                &leaf_world_bounds_map,
            );
            for (node_index, src_node) in &delta.node_writes {
                let node_index_usize = *node_index as usize;
                if self.active_config.frame_data.region_bvh_nodes.len() <= node_index_usize {
                    self.active_config
                        .frame_data
                        .region_bvh_nodes
                        .resize(node_index_usize + 1, GpuVoxelChunkBvhNode::empty());
                }
                let (world_min, world_max) = computed_node_bounds
                    .get(node_index)
                    .copied()
                    .unwrap_or(([0.0; 4], [0.0; 4]));
                self.active_config.frame_data.region_bvh_nodes[node_index_usize] =
                    Self::encode_bvh_node(src_node, world_min, world_max);
                Self::mark_dirty_index(&mut dirty.region_bvh_nodes, node_index_usize);
            }

            current_root = delta.new_root;
        }

        Self::mark_appended_tail(
            &mut dirty.chunk_headers,
            old_chunk_headers_len,
            self.active_config.frame_data.chunk_headers.len(),
        );
        Self::mark_appended_tail(
            &mut dirty.occupancy_words,
            old_occupancy_words_len,
            self.active_config.frame_data.occupancy_words.len(),
        );
        Self::mark_appended_tail(
            &mut dirty.material_words,
            old_material_words_len,
            self.active_config.frame_data.material_words.len(),
        );
        Self::mark_appended_tail(
            &mut dirty.orientation_words,
            old_orientation_words_len,
            self.active_config.frame_data.orientation_words.len(),
        );
        Self::mark_appended_tail(
            &mut dirty.macro_words,
            old_macro_words_len,
            self.active_config.frame_data.macro_words.len(),
        );
        let mutation_base_generation = self.active_config.frame_data.metadata_generation;
        self.active_config.frame_data.mutation_batch =
            Self::build_mutation_batch_from_dirty_ranges(&self.active_config.frame_data, &dirty);
        self.active_config.frame_data.mutation_base_generation = self
            .active_config
            .frame_data
            .mutation_batch
            .as_ref()
            .map(|_| mutation_base_generation);
        self.active_config.frame_data.dirty_ranges = dirty;

        self.voxel_visibility_generation = self.voxel_visibility_generation.wrapping_add(1);
        self.active_config.frame_data.metadata_generation = self.voxel_visibility_generation;
        self.active_config.frame_data.region_bvh_root_index = current_root
            .unwrap_or(higher_dimension_playground::render::VTE_REGION_BVH_INVALID_NODE);
        self.voxel_cached_visibility_bounds = Some(bounds);

        Ok(())
    }
}
