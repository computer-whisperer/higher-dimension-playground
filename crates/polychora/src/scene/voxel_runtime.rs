use super::*;
use polychora::content_registry::MaterialResolver;
use polychora::shared::chunk_payload::{ChunkPayload, ResolvedChunkPayload};
use polychora::shared::spatial::{Aabb4i, ChunkCoord, fixed_from_lattice};
use polychora::shared::voxel::BlockData;
use polychora::shared::render_tree::{
    RenderBvh, RenderBvhChunkMutationDelta, RenderBvhNodeKind, RenderLeaf, RenderLeafKind,
};

/// Convert a `[ChunkCoord; 4]` to `[i32; 4]` for GPU boundary types.
#[inline]
fn chunk_coord_to_i32(c: [ChunkCoord; 4]) -> [i32; 4] {
    [c[0].to_num::<i32>(), c[1].to_num::<i32>(), c[2].to_num::<i32>(), c[3].to_num::<i32>()]
}

/// Compute world-space AABB for a leaf's bounds at a given scale.
///
/// Bounds are already world-space half-open `[min, max)`.
/// Convert fixed-point to f32 for GPU consumption.
fn leaf_world_bounds(bounds: &polychora::shared::spatial::Aabb4, _scale_exp: i8) -> ([f32; 4], [f32; 4]) {
    (
        bounds.min.map(|v| v.to_num::<f32>()),
        bounds.max.map(|v| v.to_num::<f32>()),
    )
}

/// Pack a scale_exp (i8) into the upper 8 bits (24-31) of a u32.
/// Used for the leaf header's `uniformOrientation` field (traversal grid scale).
#[inline]
fn pack_scale_exp_into_orientation(orientation: u32, scale_exp: i8) -> u32 {
    (orientation & 0x00FFFFFFu32) | ((scale_exp as u8 as u32) << 24)
}

/// Pack tesseract orientation (9 bits, 0-383) and scale_exp (6 bits, signed)
/// into a 15-bit value for per-voxel and uniform chunk entry storage.
///
/// Layout: bits 0-8 = orientation, bits 9-14 = scale_exp (6-bit signed, range [-32, +31]).
/// Bit 15 is always 0 so the value fits in 15 bits (required for uniform chunk entries
/// where bit 15 of the u16 slot shares the UNIFORM_FLAG at bit 31 of the u32).
#[inline]
fn pack_orientation_scale_u16(orientation: u16, scale_exp: i8) -> u16 {
    let ori = orientation & 0x1FF; // 9 bits
    let scale_bits = (scale_exp as u8 as u16) & 0x3F; // 6 bits
    ori | (scale_bits << 9)
}

use std::time::Instant;

fn pack_dense_materials_words(
    dense_palette_indices: &[u16],
    block_palette: &[BlockData],
    occupancy_words: &mut [u32; OCCUPANCY_WORDS_PER_CHUNK],
    material_words: &mut [u32; MATERIAL_WORDS_PER_CHUNK],
    orientation_words: &mut [u32; ORIENTATION_WORDS_PER_CHUNK],
    macro_words: &mut [u32; MACRO_WORDS_PER_CHUNK],
    resolver: &MaterialResolver,
) -> (u32, bool, [i32; 4], [i32; 4]) {
    debug_assert_eq!(dense_palette_indices.len(), CHUNK_VOLUME);

    occupancy_words.fill(0);
    material_words.fill(0);
    orientation_words.fill(0);
    macro_words.fill(0);

    let mut solid_count = 0u32;
    let mut solid_local_min = [i32::MAX; 4];
    let mut solid_local_max = [i32::MIN; 4];

    for (voxel_idx, &palette_idx) in dense_palette_indices.iter().enumerate() {
        let block = block_palette
            .get(palette_idx as usize)
            .cloned()
            .unwrap_or(BlockData::AIR);
        let mat_u16 = resolver.resolve_block(block.namespace, block.block_type);

        let mat_word_idx = voxel_idx / 2;
        let mat_shift = ((voxel_idx & 1) * 16) as u32;
        material_words[mat_word_idx] &= !(0xFFFFu32 << mat_shift);
        material_words[mat_word_idx] |= u32::from(mat_u16) << mat_shift;

        // Pack orientation + scale into u16: bits 0-8 orientation, bits 9-15 scale_exp
        let orient_scale_u16 = pack_orientation_scale_u16(block.orientation.0, block.scale_exp);
        orientation_words[mat_word_idx] &= !(0xFFFFu32 << mat_shift);
        orientation_words[mat_word_idx] |= u32::from(orient_scale_u16) << mat_shift;

        if mat_u16 == 0 {
            continue;
        }

        solid_count = solid_count.saturating_add(1);
        let word_idx = voxel_idx / 32;
        occupancy_words[word_idx] |= 1u32 << (voxel_idx % 32);

        let x = voxel_idx % CHUNK_SIZE;
        let y = (voxel_idx / CHUNK_SIZE) % CHUNK_SIZE;
        let z = (voxel_idx / (CHUNK_SIZE * CHUNK_SIZE)) % CHUNK_SIZE;
        let w = voxel_idx / (CHUNK_SIZE * CHUNK_SIZE * CHUNK_SIZE);
        let local = [x as i32, y as i32, z as i32, w as i32];
        for axis in 0..4 {
            solid_local_min[axis] = solid_local_min[axis].min(local[axis]);
            solid_local_max[axis] = solid_local_max[axis].max(local[axis]);
        }

        let mx = x >> 1;
        let my = y >> 1;
        let mz = z >> 1;
        let mw = w >> 1;
        let macro_idx = (((mw * MACRO_CELLS_PER_AXIS + mz) * MACRO_CELLS_PER_AXIS + my)
            * MACRO_CELLS_PER_AXIS)
            + mx;
        let macro_word_idx = macro_idx / 32;
        macro_words[macro_word_idx] |= 1u32 << (macro_idx % 32);
    }

    let is_full = solid_count == CHUNK_VOLUME as u32;
    let solid_local_min = if solid_count == 0 {
        [0, 0, 0, 0]
    } else {
        solid_local_min
    };
    let solid_local_max = if solid_count == 0 {
        [0, 0, 0, 0]
    } else {
        solid_local_max
    };

    (solid_count, is_full, solid_local_min, solid_local_max)
}

impl Scene {
    fn summarize_chunk_payload_compact(resolved: Option<&ResolvedChunkPayload>) -> String {
        let Some(resolved) = resolved else {
            return "None".to_string();
        };
        match &resolved.payload {
            ChunkPayload::Empty => "Empty".to_string(),
            ChunkPayload::Uniform(idx) => {
                let block = resolved.block_palette.get(*idx as usize);
                match block {
                    Some(b) if b.is_air() => "Uniform(air)".to_string(),
                    Some(b) => format!("Uniform({}:{})", b.namespace, b.block_type),
                    None => format!("Uniform(idx={})", idx),
                }
            }
            ChunkPayload::Dense16 { materials } => {
                let non_empty = materials.iter().filter(|idx| {
                    resolved.block_palette.get(**idx as usize).map(|b| !b.is_air()).unwrap_or(false)
                }).count();
                format!("Dense16(nz={non_empty})")
            }
            ChunkPayload::PalettePacked {
                palette, bit_width, ..
            } => format!(
                "PalettePacked(palette={},bits={})",
                palette.len(),
                bit_width
            ),
        }
    }

    fn summarize_chunk_payload_list_compact(payloads: &[ResolvedChunkPayload]) -> String {
        if payloads.is_empty() {
            return "[]".to_string();
        }
        let mut out = String::from("[");
        for (idx, resolved) in payloads.iter().take(3).enumerate() {
            if idx > 0 {
                out.push_str(", ");
            }
            out.push_str(&Self::summarize_chunk_payload_compact(Some(resolved)));
        }
        if payloads.len() > 3 {
            out.push_str(", ...");
        }
        out.push(']');
        out
    }

    fn voxel_frame_root_is_valid(&self) -> bool {
        let root = self.voxel_frame_data.region_bvh_root_index;
        root != higher_dimension_playground::render::VTE_REGION_BVH_INVALID_NODE
            && (root as usize) < self.voxel_frame_data.region_bvh_nodes.len()
    }

    fn log_voxel_rebuild_failure_once(
        &mut self,
        bounds: Aabb4i,
        reason: &str,
        node_count: usize,
        leaf_count: usize,
    ) {
        let signature = (bounds, node_count, leaf_count);
        if self.voxel_last_rebuild_failure_signature == Some(signature) {
            return;
        }
        self.voxel_last_rebuild_failure_signature = Some(signature);
        eprintln!(
            "[vte-voxel-rebuild-failure] reason={} bounds={:?}->{:?} nodes={} leaves={} (preserving last-good GPU voxel frame)",
            reason,
            bounds.min,
            bounds.max,
            node_count,
            leaf_count,
        );
    }

    fn log_voxel_snapshot_rebuild(
        &self,
        bounds: Aabb4i,
        reason: &str,
        applied_deltas: usize,
        pending_deltas: usize,
        frame_root: u32,
        cpu_root: u32,
    ) {
        eprintln!(
            "[vte-voxel-snapshot-rebuild] reason={} bounds={:?}->{:?} applied_deltas={} pending_deltas={} frame_root={} cpu_root={}",
            reason,
            bounds.min,
            bounds.max,
            applied_deltas,
            pending_deltas,
            frame_root,
            cpu_root,
        );
    }

    fn camera_chunk_key(cam_pos: [f32; 4]) -> [i32; 4] {
        let cs = CHUNK_SIZE as i32;
        [
            (cam_pos[0].floor() as i32).div_euclid(cs),
            (cam_pos[1].floor() as i32).div_euclid(cs),
            (cam_pos[2].floor() as i32).div_euclid(cs),
            (cam_pos[3].floor() as i32).div_euclid(cs),
        ]
    }

    fn encode_bvh_node(
        node: &polychora::shared::render_tree::RenderBvhNode,
        world_min: [f32; 4],
        world_max: [f32; 4],
    ) -> GpuVoxelChunkBvhNode {
        let (left_child, right_child, leaf_index, flags) = match node.kind {
            RenderBvhNodeKind::Internal { left, right } => (left, right, u32::MAX, 0),
            RenderBvhNodeKind::Leaf { leaf_index } => (
                higher_dimension_playground::render::VTE_REGION_BVH_INVALID_NODE,
                higher_dimension_playground::render::VTE_REGION_BVH_INVALID_NODE,
                leaf_index,
                higher_dimension_playground::render::VTE_REGION_BVH_NODE_FLAG_LEAF,
            ),
        };
        GpuVoxelChunkBvhNode {
            world_min,
            world_max,
            left_child,
            right_child,
            leaf_index,
            flags,
        }
    }

    /// Compute float world-space bounds for all BVH nodes in a delta write set.
    ///
    /// This handles arbitrary index ordering (freed node IDs can be reused in any
    /// order by the allocator) by recursing into children as needed. Each child's
    /// bounds are resolved from: (1) already-computed delta nodes, (2) pre-existing
    /// GPU array entries, or (3) recursively computed from the delta write set.
    fn compute_all_delta_node_world_bounds(
        node_writes: &[(u32, polychora::shared::render_tree::RenderBvhNode)],
        gpu_nodes: &[GpuVoxelChunkBvhNode],
        leaf_headers: &[GpuVoxelLeafHeader],
        leaf_world_bounds_map: &std::collections::HashMap<u32, ([f32; 4], [f32; 4])>,
    ) -> std::collections::HashMap<u32, ([f32; 4], [f32; 4])> {
        // Index delta nodes by their slot ID for O(1) lookup during recursion.
        let delta_by_id: std::collections::HashMap<u32, &polychora::shared::render_tree::RenderBvhNode> =
            node_writes.iter().map(|(id, node)| (*id, node)).collect();
        let mut computed = std::collections::HashMap::<u32, ([f32; 4], [f32; 4])>::new();

        for &(node_id, _) in node_writes {
            Self::resolve_node_world_bounds(
                node_id,
                &delta_by_id,
                gpu_nodes,
                leaf_headers,
                leaf_world_bounds_map,
                &mut computed,
            );
        }
        computed
    }

    fn resolve_node_world_bounds(
        node_id: u32,
        delta_by_id: &std::collections::HashMap<u32, &polychora::shared::render_tree::RenderBvhNode>,
        gpu_nodes: &[GpuVoxelChunkBvhNode],
        leaf_headers: &[GpuVoxelLeafHeader],
        leaf_world_bounds_map: &std::collections::HashMap<u32, ([f32; 4], [f32; 4])>,
        computed: &mut std::collections::HashMap<u32, ([f32; 4], [f32; 4])>,
    ) -> ([f32; 4], [f32; 4]) {
        if let Some(&bounds) = computed.get(&node_id) {
            return bounds;
        }
        let Some(node) = delta_by_id.get(&node_id) else {
            // Not in delta — read from pre-existing GPU array.
            return gpu_nodes.get(node_id as usize)
                .map(|n| (n.world_min, n.world_max))
                .unwrap_or(([0.0; 4], [0.0; 4]));
        };
        let bounds = match node.kind {
            RenderBvhNodeKind::Leaf { leaf_index } => {
                if let Some(&b) = leaf_world_bounds_map.get(&leaf_index) {
                    b
                } else {
                    // Pre-existing leaf: extract scale_exp from the packed
                    // uniform_orientation field in the existing leaf header.
                    let scale_exp = leaf_headers
                        .get(leaf_index as usize)
                        .map(|lh| (lh.uniform_orientation >> 24) as i8)
                        .unwrap_or(0);
                    leaf_world_bounds(&node.bounds, scale_exp)
                }
            }
            RenderBvhNodeKind::Internal { left, right } => {
                let (lmin, lmax) = Self::resolve_node_world_bounds(
                    left, delta_by_id, gpu_nodes, leaf_headers,
                    leaf_world_bounds_map, computed,
                );
                let (rmin, rmax) = Self::resolve_node_world_bounds(
                    right, delta_by_id, gpu_nodes, leaf_headers,
                    leaf_world_bounds_map, computed,
                );
                ([
                    lmin[0].min(rmin[0]), lmin[1].min(rmin[1]),
                    lmin[2].min(rmin[2]), lmin[3].min(rmin[3]),
                ], [
                    lmax[0].max(rmax[0]), lmax[1].max(rmax[1]),
                    lmax[2].max(rmax[2]), lmax[3].max(rmax[3]),
                ])
            }
        };
        computed.insert(node_id, bounds);
        bounds
    }

    fn append_dense_payload_encoded(
        voxel_frame_data: &mut VoxelFrameData,
        dense_payload_cache: &mut std::collections::HashMap<ChunkPayload, u32>,
        payload: &ChunkPayload,
        block_palette: &[BlockData],
        resolver: &MaterialResolver,
    ) -> Result<u32, String> {
        if let Some(&encoded) = dense_payload_cache.get(payload) {
            return Ok(encoded);
        }
        let dense_palette_indices = payload
            .dense_materials()
            .map_err(|error| format!("dense payload decode failed: {error}"))?;
        if dense_palette_indices.len() != CHUNK_VOLUME {
            return Err(format!(
                "dense payload voxel count {} != {}",
                dense_palette_indices.len(),
                CHUNK_VOLUME
            ));
        }
        let mut occ = [0u32; OCCUPANCY_WORDS_PER_CHUNK];
        let mut mat = [0u32; MATERIAL_WORDS_PER_CHUNK];
        let mut ori = [0u32; ORIENTATION_WORDS_PER_CHUNK];
        let mut mac = [0u32; MACRO_WORDS_PER_CHUNK];
        let (_solid_count, is_full, solid_local_min, solid_local_max) =
            pack_dense_materials_words(&dense_palette_indices, block_palette, &mut occ, &mut mat, &mut ori, &mut mac, resolver);
        let chunk_index = voxel_frame_data.chunk_headers.len() as u32;
        let occ_offset = voxel_frame_data.occupancy_words.len() as u32;
        let mat_offset = voxel_frame_data.material_words.len() as u32;
        let ori_offset = voxel_frame_data.orientation_words.len() as u32;
        let mac_offset = voxel_frame_data.macro_words.len() as u32;
        voxel_frame_data.occupancy_words.extend_from_slice(&occ);
        voxel_frame_data.material_words.extend_from_slice(&mat);
        voxel_frame_data.orientation_words.extend_from_slice(&ori);
        voxel_frame_data.macro_words.extend_from_slice(&mac);
        let mut flags = 0u32;
        if is_full {
            flags |= GpuVoxelChunkHeader::FLAG_FULL;
        }
        voxel_frame_data.chunk_headers.push(GpuVoxelChunkHeader {
            occupancy_word_offset: occ_offset,
            material_word_offset: mat_offset,
            flags,
            macro_word_offset: mac_offset,
            solid_local_min,
            solid_local_max,
            orientation_word_offset: ori_offset,
            _padding: 0,
        });
        let encoded = chunk_index.saturating_add(1);
        dense_payload_cache.insert(payload.clone(), encoded);
        Ok(encoded)
    }

    fn invalid_leaf_header() -> GpuVoxelLeafHeader {
        GpuVoxelLeafHeader {
            min_chunk_coord: [0, 0, 0, 0],
            max_chunk_coord: [-1, -1, -1, -1],
            leaf_kind: higher_dimension_playground::render::VTE_LEAF_KIND_UNIFORM,
            uniform_material: 0,
            chunk_entry_offset: 0,
            uniform_orientation: 0,
        }
    }

    fn mark_dirty_range(slot: &mut Option<std::ops::Range<usize>>, range: std::ops::Range<usize>) {
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

    fn mark_dirty_index(slot: &mut Option<std::ops::Range<usize>>, index: usize) {
        Self::mark_dirty_range(slot, index..index.saturating_add(1));
    }

    fn mark_appended_tail(
        slot: &mut Option<std::ops::Range<usize>>,
        old_len: usize,
        new_len: usize,
    ) {
        if new_len > old_len {
            Self::mark_dirty_range(slot, old_len..new_len);
        }
    }

    fn full_dirty_ranges_from_frame(voxel_frame_data: &VoxelFrameData) -> VoxelFrameDirtyRanges {
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

    fn build_mutation_batch_from_dirty_ranges(
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

    fn encode_leaf_chunk_entries(
        voxel_frame_data: &mut VoxelFrameData,
        dense_payload_cache: &mut std::collections::HashMap<ChunkPayload, u32>,
        leaf: &RenderLeaf,
        resolver: &MaterialResolver,
    ) -> Result<Vec<u32>, String> {
        let RenderLeafKind::VoxelChunkArray(chunk_array) = &leaf.kind else {
            return Ok(Vec::new());
        };
        let scale_exp = chunk_array.scale_exp;
        let Some(leaf_extents) = leaf.bounds.chunk_extents_at_scale(scale_exp) else {
            return Ok(Vec::new());
        };
        // Leaf bounds must be grid-aligned at scale_exp. Misaligned bounds
        // produce empty entries (the chunk world-origin falls outside the
        // bounds). The render tree guarantees alignment; assert it here so
        // regressions are caught in development.
        debug_assert_eq!(
            Aabb4i::from_lattice_bounds(
                leaf.bounds.to_chunk_lattice_bounds(scale_exp).0,
                leaf.bounds.to_chunk_lattice_bounds(scale_exp).1,
                scale_exp,
            ),
            leaf.bounds,
            "encode_leaf_chunk_entries: leaf bounds misaligned at scale_exp={}",
            scale_exp,
        );
        let src_indices = chunk_array
            .decode_dense_indices()
            .map_err(|error| format!("decode chunk-array leaf failed: {error:?}"))?;
        let src_dims = chunk_array
            .bounds
            .chunk_extents_at_scale(scale_exp)
            .ok_or_else(|| "chunk-array source extents missing".to_string())?;

        let (leaf_lattice_min, _leaf_lattice_max) = leaf.bounds.to_chunk_lattice_bounds(scale_exp);
        let (src_lattice_min, _src_lattice_max) = chunk_array.bounds.to_chunk_lattice_bounds(scale_exp);

        let mut encoded = Vec::<u32>::with_capacity(leaf.bounds.chunk_cell_count_at_scale(scale_exp).unwrap_or(0));
        for w in 0..leaf_extents[3] {
            for z in 0..leaf_extents[2] {
                for y in 0..leaf_extents[1] {
                    for x in 0..leaf_extents[0] {
                        // Lattice-space coordinate for this cell
                        let lat = [
                            leaf_lattice_min[0] + x as i32,
                            leaf_lattice_min[1] + y as i32,
                            leaf_lattice_min[2] + z as i32,
                            leaf_lattice_min[3] + w as i32,
                        ];
                        // Convert to fixed-point to check against source bounds
                        let chunk_coord = [
                            fixed_from_lattice(lat[0], scale_exp),
                            fixed_from_lattice(lat[1], scale_exp),
                            fixed_from_lattice(lat[2], scale_exp),
                            fixed_from_lattice(lat[3], scale_exp),
                        ];
                        let lx = (lat[0] - src_lattice_min[0]) as usize;
                        let ly = (lat[1] - src_lattice_min[1]) as usize;
                        let lz = (lat[2] - src_lattice_min[2]) as usize;
                        let lw = (lat[3] - src_lattice_min[3]) as usize;
                        debug_assert!(
                            chunk_array.bounds.contains_chunk_world_min(chunk_coord),
                            "chunk at lattice {:?} (world {:?}) outside source bounds {:?}",
                            lat, chunk_coord, chunk_array.bounds,
                        );
                        let linear =
                            lx + src_dims[0] * (ly + src_dims[1] * (lz + src_dims[2] * lw));
                        let palette_index = src_indices
                            .get(linear)
                            .copied()
                            .ok_or_else(|| "chunk-array index out of bounds".to_string())?
                            as usize;
                        let payload = chunk_array
                            .chunk_palette
                            .get(palette_index)
                            .ok_or_else(|| "chunk-array palette out of bounds".to_string())?;
                        let encoded_entry = match payload {
                            ChunkPayload::Empty => {
                                higher_dimension_playground::render::VTE_LEAF_CHUNK_ENTRY_EMPTY
                            }
                            ChunkPayload::Uniform(idx) => {
                                let block = chunk_array.block_palette.get(*idx as usize).cloned().unwrap_or(BlockData::AIR);
                                let mat = resolver.resolve_block(block.namespace, block.block_type);
                                if mat == 0 {
                                    higher_dimension_playground::render::VTE_LEAF_CHUNK_ENTRY_EMPTY
                                } else {
                                    higher_dimension_playground::render::VTE_LEAF_CHUNK_ENTRY_UNIFORM_FLAG
                                        | (u32::from(pack_orientation_scale_u16(block.orientation.0, block.scale_exp)) << 16)
                                        | u32::from(mat)
                                }
                            }
                            dense_payload => Self::append_dense_payload_encoded(
                                voxel_frame_data,
                                dense_payload_cache,
                                dense_payload,
                                &chunk_array.block_palette,
                                resolver,
                            )?,
                        };
                        encoded.push(encoded_entry);
                    }
                }
            }
        }
        Ok(encoded)
    }

    fn build_leaf_entry_spans_from_headers(
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
                let span = header.max_chunk_coord[axis] as i64
                    - header.min_chunk_coord[axis] as i64
                    + 1;
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

    fn sync_leaf_entry_allocator_from_frame(&mut self) {
        let spans = Self::build_leaf_entry_spans_from_headers(&self.voxel_frame_data);
        let free_spans = Self::build_free_spans_from_leaf_spans(
            &spans,
            self.voxel_frame_data.leaf_chunk_entries.len(),
        );
        self.voxel_leaf_entry_spans = spans;
        self.voxel_leaf_entry_free_spans = free_spans;
    }

    fn merge_free_span(
        free_spans: &mut Vec<std::ops::Range<usize>>,
        new_span: std::ops::Range<usize>,
    ) {
        if new_span.start >= new_span.end {
            return;
        }
        free_spans.push(new_span);
        free_spans.sort_unstable_by_key(|span| span.start);
        let mut merged = Vec::<std::ops::Range<usize>>::with_capacity(free_spans.len());
        for span in free_spans.drain(..) {
            if let Some(last) = merged.last_mut() {
                if span.start <= last.end {
                    last.end = last.end.max(span.end);
                } else {
                    merged.push(span);
                }
            } else {
                merged.push(span);
            }
        }
        *free_spans = merged;
    }

    fn build_free_spans_from_leaf_spans(
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

    fn allocate_leaf_entry_span(
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

    fn apply_render_bvh_mutation_deltas_to_voxel_frame_data(
        &mut self,
        deltas: &[RenderBvhChunkMutationDelta],
        bounds: Aabb4i,
        resolver: &MaterialResolver,
    ) -> Result<(), String> {
        if !bounds.is_valid() {
            return Ok(());
        }
        if self.voxel_leaf_entry_spans.len() != self.voxel_frame_data.leaf_headers.len() {
            self.sync_leaf_entry_allocator_from_frame();
        }
        let leaf_spans = &mut self.voxel_leaf_entry_spans;
        let free_spans = &mut self.voxel_leaf_entry_free_spans;
        let old_chunk_headers_len = self.voxel_frame_data.chunk_headers.len();
        let old_occupancy_words_len = self.voxel_frame_data.occupancy_words.len();
        let old_material_words_len = self.voxel_frame_data.material_words.len();
        let old_orientation_words_len = self.voxel_frame_data.orientation_words.len();
        let old_macro_words_len = self.voxel_frame_data.macro_words.len();
        let mut dirty = VoxelFrameDirtyRanges::default();
        let mut current_root = if self.voxel_frame_data.region_bvh_root_index
            == higher_dimension_playground::render::VTE_REGION_BVH_INVALID_NODE
        {
            None
        } else {
            Some(self.voxel_frame_data.region_bvh_root_index)
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
                if node_index < self.voxel_frame_data.region_bvh_nodes.len() {
                    self.voxel_frame_data.region_bvh_nodes[node_index] =
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
                if leaf_index < self.voxel_frame_data.leaf_headers.len() {
                    self.voxel_frame_data.leaf_headers[leaf_index] = Self::invalid_leaf_header();
                    Self::mark_dirty_index(&mut dirty.leaf_headers, leaf_index);
                }
            }

            for (leaf_index, leaf) in &delta.leaf_writes {
                let leaf_index = *leaf_index as usize;
                if self.voxel_frame_data.leaf_headers.len() <= leaf_index {
                    self.voxel_frame_data
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
                        self.voxel_frame_data.leaf_headers[leaf_index] = GpuVoxelLeafHeader {
                            min_chunk_coord: lat_min,
                            max_chunk_coord: lat_max,
                            leaf_kind: higher_dimension_playground::render::VTE_LEAF_KIND_UNIFORM,
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
                            &mut self.voxel_frame_data,
                            &mut self.voxel_dense_payload_encoded_cache,
                            leaf,
                            resolver,
                        )?;
                        let span = Self::allocate_leaf_entry_span(
                            free_spans,
                            &mut self.voxel_frame_data.leaf_chunk_entries,
                            entries.len(),
                        )?;
                        self.voxel_frame_data.leaf_chunk_entries[span.clone()]
                            .copy_from_slice(entries.as_slice());
                        Self::mark_dirty_range(&mut dirty.leaf_chunk_entries, span.clone());
                        leaf_spans[leaf_index] = Some(span.clone());
                        self.voxel_frame_data.leaf_headers[leaf_index] = GpuVoxelLeafHeader {
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
            let mut leaf_world_bounds_map = std::collections::HashMap::<u32, ([f32; 4], [f32; 4])>::new();
            for (leaf_index, leaf) in &delta.leaf_writes {
                let scale_exp = match &leaf.kind {
                    RenderLeafKind::Uniform(block) => block.scale_exp,
                    RenderLeafKind::VoxelChunkArray(ca) => ca.scale_exp,
                };
                leaf_world_bounds_map.insert(*leaf_index, leaf_world_bounds(&leaf.bounds, scale_exp));
            }

            // Write BVH nodes after leaves so we can compute world bounds.
            // Pre-compute all node bounds recursively so that internal nodes
            // resolve their children correctly regardless of index order (freed
            // node IDs can be reused in arbitrary order by the allocator).
            let computed_node_bounds = Self::compute_all_delta_node_world_bounds(
                &delta.node_writes,
                &self.voxel_frame_data.region_bvh_nodes,
                &self.voxel_frame_data.leaf_headers,
                &leaf_world_bounds_map,
            );
            for (node_index, src_node) in &delta.node_writes {
                let node_index_usize = *node_index as usize;
                if self.voxel_frame_data.region_bvh_nodes.len() <= node_index_usize {
                    self.voxel_frame_data
                        .region_bvh_nodes
                        .resize(node_index_usize + 1, GpuVoxelChunkBvhNode::empty());
                }
                let (world_min, world_max) = computed_node_bounds
                    .get(node_index)
                    .copied()
                    .unwrap_or(([0.0; 4], [0.0; 4]));
                self.voxel_frame_data.region_bvh_nodes[node_index_usize] =
                    Self::encode_bvh_node(src_node, world_min, world_max);
                Self::mark_dirty_index(&mut dirty.region_bvh_nodes, node_index_usize);
            }

            current_root = delta.new_root;
        }

        Self::mark_appended_tail(
            &mut dirty.chunk_headers,
            old_chunk_headers_len,
            self.voxel_frame_data.chunk_headers.len(),
        );
        Self::mark_appended_tail(
            &mut dirty.occupancy_words,
            old_occupancy_words_len,
            self.voxel_frame_data.occupancy_words.len(),
        );
        Self::mark_appended_tail(
            &mut dirty.material_words,
            old_material_words_len,
            self.voxel_frame_data.material_words.len(),
        );
        Self::mark_appended_tail(
            &mut dirty.orientation_words,
            old_orientation_words_len,
            self.voxel_frame_data.orientation_words.len(),
        );
        Self::mark_appended_tail(
            &mut dirty.macro_words,
            old_macro_words_len,
            self.voxel_frame_data.macro_words.len(),
        );
        let mutation_base_generation = self.voxel_frame_data.metadata_generation;
        self.voxel_frame_data.mutation_batch =
            Self::build_mutation_batch_from_dirty_ranges(&self.voxel_frame_data, &dirty);
        self.voxel_frame_data.mutation_base_generation =
            self.voxel_frame_data
                .mutation_batch
                .as_ref()
                .map(|_| mutation_base_generation);
        self.voxel_frame_data.dirty_ranges = dirty;

        self.voxel_visibility_generation = self.voxel_visibility_generation.wrapping_add(1);
        self.voxel_frame_data.metadata_generation = self.voxel_visibility_generation;
        self.voxel_frame_data.region_bvh_root_index = current_root
            .unwrap_or(higher_dimension_playground::render::VTE_REGION_BVH_INVALID_NODE);
        self.voxel_cached_visibility_bounds = Some(bounds);
        self.voxel_last_rebuild_failure_signature = None;
        Ok(())
    }

    fn build_voxel_frame_buffers_from_render_bvh(
        render_bvh: &RenderBvh,
        resolver: &MaterialResolver,
    ) -> Option<VoxelFrameDataBuffers> {
        let mut dense_chunk_headers = Vec::<GpuVoxelChunkHeader>::new();
        let mut occupancy_words = Vec::<u32>::new();
        let mut material_words = Vec::<u32>::new();
        let mut orientation_words = Vec::<u32>::new();
        let mut macro_words = Vec::<u32>::new();
        let mut leaf_headers = Vec::<GpuVoxelLeafHeader>::with_capacity(render_bvh.leaves.len());
        let mut leaf_chunk_entries = Vec::<u32>::new();
        let mut region_bvh_nodes =
            Vec::<GpuVoxelChunkBvhNode>::with_capacity(render_bvh.nodes.len());
        let mut dense_payload_encoded_cache = std::collections::HashMap::<ChunkPayload, u32>::new();

        // Pre-compute leaf world bounds for BVH node encoding.
        let mut per_leaf_world_bounds = Vec::<([f32; 4], [f32; 4])>::with_capacity(render_bvh.leaves.len());

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
                    leaf_headers.push(GpuVoxelLeafHeader {
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
                RenderLeafKind::VoxelChunkArray(chunk_array) => {
                    let (lat_min, lat_max) = leaf.bounds.to_chunk_lattice_bounds(scale_exp);
                    let Some(leaf_extents) = leaf.bounds.chunk_extents_at_scale(scale_exp) else {
                        leaf_headers.push(GpuVoxelLeafHeader {
                            min_chunk_coord: lat_min,
                            max_chunk_coord: lat_max,
                            leaf_kind: higher_dimension_playground::render::VTE_LEAF_KIND_VOXEL_CHUNK_ARRAY,
                            uniform_material: 0,
                            chunk_entry_offset: 0,
                            uniform_orientation: pack_scale_exp_into_orientation(0, scale_exp),
                        });
                        continue;
                    };
                    let leaf_cell_count = leaf.bounds.chunk_cell_count_at_scale(scale_exp).unwrap_or(0);
                    let entry_offset = leaf_chunk_entries.len() as u32;
                    leaf_chunk_entries.reserve(leaf_cell_count);

                    let src_indices = chunk_array.decode_dense_indices().ok();
                    let src_dims = chunk_array.bounds.chunk_extents_at_scale(scale_exp);
                    let (src_lattice_min, _src_lattice_max) = chunk_array.bounds.to_chunk_lattice_bounds(scale_exp);
                    let mut palette_encoded_cache =
                        vec![None::<u32>; chunk_array.chunk_palette.len()];

                    for w in 0..leaf_extents[3] {
                        for z in 0..leaf_extents[2] {
                            for y in 0..leaf_extents[1] {
                                for x in 0..leaf_extents[0] {
                                    let lat = [
                                        lat_min[0] + x as i32,
                                        lat_min[1] + y as i32,
                                        lat_min[2] + z as i32,
                                        lat_min[3] + w as i32,
                                    ];
                                    let chunk_coord = [
                                        fixed_from_lattice(lat[0], scale_exp),
                                        fixed_from_lattice(lat[1], scale_exp),
                                        fixed_from_lattice(lat[2], scale_exp),
                                        fixed_from_lattice(lat[3], scale_exp),
                                    ];
                                    let mut encoded = higher_dimension_playground::render::VTE_LEAF_CHUNK_ENTRY_EMPTY;
                                    if let (Some(indices), Some(src_dims)) =
                                        (src_indices.as_ref(), src_dims)
                                    {
                                        if chunk_array.bounds.contains_chunk_world_min(chunk_coord)
                                        {
                                            let lx = (lat[0] - src_lattice_min[0]) as usize;
                                            let ly = (lat[1] - src_lattice_min[1]) as usize;
                                            let lz = (lat[2] - src_lattice_min[2]) as usize;
                                            let lw = (lat[3] - src_lattice_min[3]) as usize;
                                            let linear = lx
                                                + src_dims[0]
                                                    * (ly + src_dims[1] * (lz + src_dims[2] * lw));
                                            if let Some(&palette_index) = indices.get(linear) {
                                                let palette_index = palette_index as usize;
                                                if let Some(Some(cached_encoded)) =
                                                    palette_encoded_cache.get(palette_index)
                                                {
                                                    encoded = *cached_encoded;
                                                } else if let Some(payload) =
                                                    chunk_array.chunk_palette.get(palette_index)
                                                {
                                                    let resolved_encoded = match payload {
                                                        ChunkPayload::Empty => {
                                                            higher_dimension_playground::render::VTE_LEAF_CHUNK_ENTRY_EMPTY
                                                        }
                                                        ChunkPayload::Uniform(idx) => {
                                                            let block = chunk_array.block_palette.get(*idx as usize).cloned().unwrap_or(BlockData::AIR);
                                                            let mat = resolver.resolve_block(block.namespace, block.block_type);
                                                            if mat == 0 {
                                                                higher_dimension_playground::render::VTE_LEAF_CHUNK_ENTRY_EMPTY
                                                            } else {
                                                                higher_dimension_playground::render::VTE_LEAF_CHUNK_ENTRY_UNIFORM_FLAG
                                                                    | (u32::from(pack_orientation_scale_u16(block.orientation.0, block.scale_exp)) << 16)
                                                                    | u32::from(mat)
                                                            }
                                                        }
                                                        _ => {
                                                            if let Some(&cached_dense_encoded) =
                                                                dense_payload_encoded_cache
                                                                    .get(payload)
                                                            {
                                                                cached_dense_encoded
                                                            } else {
                                                                let dense_materials = match payload
                                                                    .dense_materials()
                                                                {
                                                                    Ok(m) => m,
                                                                    Err(_) => {
                                                                        Vec::new()
                                                                    }
                                                                };
                                                                if dense_materials.len()
                                                                    != CHUNK_VOLUME
                                                                {
                                                                    higher_dimension_playground::render::VTE_LEAF_CHUNK_ENTRY_EMPTY
                                                                } else {
                                                                    let mut occ = [0u32;
                                                                        OCCUPANCY_WORDS_PER_CHUNK];
                                                                    let mut mat = [0u32;
                                                                        MATERIAL_WORDS_PER_CHUNK];
                                                                    let mut ori = [0u32;
                                                                        ORIENTATION_WORDS_PER_CHUNK];
                                                                    let mut mac = [0u32;
                                                                        MACRO_WORDS_PER_CHUNK];
                                                                    let (_solid_count, is_full, solid_local_min, solid_local_max) =
                                                                        pack_dense_materials_words(
                                                                            &dense_materials,
                                                                            &chunk_array.block_palette,
                                                                            &mut occ,
                                                                            &mut mat,
                                                                            &mut ori,
                                                                            &mut mac,
                                                                            resolver,
                                                                        );
                                                                    let chunk_index =
                                                                        dense_chunk_headers.len() as u32;
                                                                    let occ_offset =
                                                                        occupancy_words.len() as u32;
                                                                    let mat_offset =
                                                                        material_words.len() as u32;
                                                                    let ori_offset =
                                                                        orientation_words.len() as u32;
                                                                    let mac_offset =
                                                                        macro_words.len() as u32;
                                                                    occupancy_words
                                                                        .extend_from_slice(&occ);
                                                                    material_words
                                                                        .extend_from_slice(&mat);
                                                                    orientation_words
                                                                        .extend_from_slice(&ori);
                                                                    macro_words
                                                                        .extend_from_slice(&mac);
                                                                    let mut flags = 0u32;
                                                                    if is_full {
                                                                        flags |=
                                                                            GpuVoxelChunkHeader::FLAG_FULL;
                                                                    }
                                                                    dense_chunk_headers.push(
                                                                        GpuVoxelChunkHeader {
                                                                            occupancy_word_offset:
                                                                                occ_offset,
                                                                            material_word_offset:
                                                                                mat_offset,
                                                                            flags,
                                                                            macro_word_offset:
                                                                                mac_offset,
                                                                            solid_local_min,
                                                                            solid_local_max,
                                                                            orientation_word_offset:
                                                                                ori_offset,
                                                                            _padding: 0,
                                                                        },
                                                                    );
                                                                    let encoded_dense =
                                                                        chunk_index.saturating_add(1);
                                                                    dense_payload_encoded_cache
                                                                        .insert(
                                                                            payload.clone(),
                                                                            encoded_dense,
                                                                        );
                                                                    encoded_dense
                                                                }
                                                            }
                                                        }
                                                    };
                                                    if let Some(cached_slot) =
                                                        palette_encoded_cache.get_mut(palette_index)
                                                    {
                                                        *cached_slot = Some(resolved_encoded);
                                                    }
                                                    encoded = resolved_encoded;
                                                }
                                            }
                                        }
                                    }
                                    leaf_chunk_entries.push(encoded);
                                }
                            }
                        }
                    }

                    leaf_headers.push(GpuVoxelLeafHeader {
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
                RenderBvhNodeKind::Leaf { leaf_index } => {
                    per_leaf_world_bounds.get(leaf_index as usize)
                        .copied()
                        .unwrap_or(leaf_world_bounds(&node.bounds, 0))
                }
                RenderBvhNodeKind::Internal { left, right } => {
                    let left_bounds = region_bvh_nodes.get(left as usize).map(|n| (n.world_min, n.world_max));
                    let right_bounds = region_bvh_nodes.get(right as usize).map(|n| (n.world_min, n.world_max));
                    match (left_bounds, right_bounds) {
                        (Some((lmin, lmax)), Some((rmin, rmax))) => {
                            ([
                                lmin[0].min(rmin[0]), lmin[1].min(rmin[1]),
                                lmin[2].min(rmin[2]), lmin[3].min(rmin[3]),
                            ], [
                                lmax[0].max(rmax[0]), lmax[1].max(rmax[1]),
                                lmax[2].max(rmax[2]), lmax[3].max(rmax[3]),
                            ])
                        }
                        (Some(b), None) | (None, Some(b)) => b,
                        (None, None) => ([0.0; 4], [0.0; 4]),
                    }
                }
            };
            region_bvh_nodes.push(Self::encode_bvh_node(node, world_min, world_max));
        }

        Some(VoxelFrameDataBuffers {
            region_bvh_root_index: render_bvh
                .root
                .unwrap_or(higher_dimension_playground::render::VTE_REGION_BVH_INVALID_NODE),
            dense_payload_encoded_cache,
            chunk_headers: dense_chunk_headers,
            occupancy_words,
            material_words,
            orientation_words,
            macro_words,
            region_bvh_nodes,
            leaf_headers,
            leaf_chunk_entries,
        })
    }

    fn clear_voxel_frame_buffers(&mut self) {
        self.voxel_frame_data.region_bvh_root_index =
            higher_dimension_playground::render::VTE_REGION_BVH_INVALID_NODE;
        self.voxel_frame_data.mutation_base_generation = None;
        self.voxel_frame_data.dirty_ranges = VoxelFrameDirtyRanges::default();
        self.voxel_frame_data.mutation_batch = None;
        self.voxel_dense_payload_encoded_cache.clear();
        self.voxel_leaf_entry_spans.clear();
        self.voxel_leaf_entry_free_spans.clear();
        self.voxel_frame_data.chunk_headers.clear();
        self.voxel_frame_data.occupancy_words.clear();
        self.voxel_frame_data.material_words.clear();
        self.voxel_frame_data.orientation_words.clear();
        self.voxel_frame_data.macro_words.clear();
        self.voxel_frame_data.region_bvh_nodes.clear();
        self.voxel_frame_data.leaf_headers.clear();
        self.voxel_frame_data.leaf_chunk_entries.clear();
    }

    fn apply_voxel_frame_buffers(
        &mut self,
        bounds: Aabb4i,
        buffers: Option<VoxelFrameDataBuffers>,
    ) {
        self.voxel_visibility_generation = self.voxel_visibility_generation.wrapping_add(1);
        self.voxel_frame_data.metadata_generation = self.voxel_visibility_generation;
        self.voxel_frame_data.mutation_base_generation = None;
        if let Some(buffers) = buffers {
            self.voxel_frame_data.region_bvh_root_index = buffers.region_bvh_root_index;
            self.voxel_dense_payload_encoded_cache = buffers.dense_payload_encoded_cache;
            self.voxel_frame_data.chunk_headers = buffers.chunk_headers;
            self.voxel_frame_data.occupancy_words = buffers.occupancy_words;
            self.voxel_frame_data.material_words = buffers.material_words;
            self.voxel_frame_data.orientation_words = buffers.orientation_words;
            self.voxel_frame_data.macro_words = buffers.macro_words;
            self.voxel_frame_data.region_bvh_nodes = buffers.region_bvh_nodes;
            self.voxel_frame_data.leaf_headers = buffers.leaf_headers;
            self.voxel_frame_data.leaf_chunk_entries = buffers.leaf_chunk_entries;
            self.voxel_frame_data.mutation_batch = None;
            self.voxel_frame_data.dirty_ranges =
                Self::full_dirty_ranges_from_frame(&self.voxel_frame_data);
            self.sync_leaf_entry_allocator_from_frame();
            self.voxel_last_rebuild_failure_signature = None;
        } else {
            self.clear_voxel_frame_buffers();
            self.voxel_last_rebuild_failure_signature = None;
        }
        self.voxel_cached_visibility_bounds = Some(bounds);
    }

    /// Build the voxel-native frame payload for VTE.
    pub fn build_voxel_frame_data(
        &mut self,
        cam_pos: [f32; 4],
        _cam_forward: [f32; 4],
        max_trace_distance: f32,
        resolver: &MaterialResolver,
    ) -> &VoxelFrameData {
        const SCENE_RESIDENCY_RADIUS_MULTIPLIER: i32 = 2;
        const SCENE_RESIDENCY_EXTRA_CHUNKS: i32 = 2;

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
        let scene_bounds = self
            .voxel_cached_visibility_bounds
            .filter(|active_bounds| Self::bounds_contains_bounds(*active_bounds, view_bounds))
            .unwrap_or(desired_scene_bounds);

        let frame_root_valid = self.voxel_frame_root_is_valid();
        let cached_render_bvh_empty = self
            .render_bvh_cache
            .as_ref()
            .map(|bvh| bvh.root.is_none())
            .unwrap_or(false);
        let has_pending_render_updates = self.voxel_pending_render_bvh_rebuild
            || !self.voxel_pending_render_bvh_mutation_deltas.is_empty();
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
                self.voxel_pending_render_bvh_rebuild,
                self.voxel_pending_render_bvh_mutation_deltas.len()
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
                    self.voxel_frame_data.region_bvh_root_index,
                    self.render_bvh_cache.as_ref().and_then(|bvh| bvh.root),
                );
            }

            if needs_rebuild {
                if let Some(render_bvh) = self.render_bvh_cache.as_ref() {
                    let cpu_root = render_bvh.root.unwrap_or(
                        higher_dimension_playground::render::VTE_REGION_BVH_INVALID_NODE,
                    );
                    self.log_voxel_snapshot_rebuild(
                        scene_bounds,
                        "pending_rebuild_flag",
                        0,
                        self.voxel_pending_render_bvh_mutation_deltas.len(),
                        self.voxel_frame_data.region_bvh_root_index,
                        cpu_root,
                    );
                    if let Some(buffers) =
                        Self::build_voxel_frame_buffers_from_render_bvh(render_bvh, resolver)
                    {
                        self.apply_voxel_frame_buffers(scene_bounds, Some(buffers));
                        if std::env::var_os("R4D_EDIT_RENDER_SYNC_DIAG").is_some() {
                            eprintln!(
                                "[edit-sync-voxel-rebuild] scene_bounds={:?}->{:?} result=ok frame_root={}",
                                scene_bounds.min,
                                scene_bounds.max,
                                self.voxel_frame_data.region_bvh_root_index
                            );
                        }
                    } else {
                        // Preserve last-good GPU voxel buffers and keep retrying full snapshot builds.
                        self.voxel_pending_render_bvh_rebuild = true;
                        self.log_voxel_rebuild_failure_once(
                            scene_bounds,
                            "snapshot_encode_capacity_overflow",
                            render_bvh.nodes.len(),
                            render_bvh.leaves.len(),
                        );
                        if std::env::var_os("R4D_EDIT_RENDER_SYNC_DIAG").is_some() {
                            eprintln!(
                                "[edit-sync-voxel-rebuild] scene_bounds={:?}->{:?} result=encode_overflow",
                                scene_bounds.min,
                                scene_bounds.max
                            );
                        }
                    }
                }
            } else if !deltas.is_empty() {
                let applied_deltas = deltas.len();
                let mut apply_ok = false;
                let mut apply_error: Option<String> = None;
                if let Some(render_bvh) = self.render_bvh_cache.take() {
                    match self
                        .apply_render_bvh_mutation_deltas_to_voxel_frame_data(&deltas, scene_bounds, resolver)
                    {
                        Ok(()) => {
                            apply_ok = true;
                        }
                        Err(error) => {
                            apply_error = Some(error);
                        }
                    }
                    self.render_bvh_cache = Some(render_bvh);
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
                    self.voxel_pending_render_bvh_mutation_deltas.clear();
                    if let Some(render_bvh) = self.render_bvh_cache.as_ref() {
                        let cpu_root = render_bvh.root.unwrap_or(
                            higher_dimension_playground::render::VTE_REGION_BVH_INVALID_NODE,
                        );
                        self.log_voxel_snapshot_rebuild(
                            scene_bounds,
                            apply_error.as_deref().unwrap_or("delta_apply_failed"),
                            applied_deltas,
                            self.voxel_pending_render_bvh_mutation_deltas.len(),
                            self.voxel_frame_data.region_bvh_root_index,
                            cpu_root,
                        );
                        if let Some(buffers) =
                            Self::build_voxel_frame_buffers_from_render_bvh(render_bvh, resolver)
                        {
                            self.apply_voxel_frame_buffers(scene_bounds, Some(buffers));
                        } else {
                            self.voxel_pending_render_bvh_rebuild = true;
                            self.log_voxel_rebuild_failure_once(
                                scene_bounds,
                                "delta_recovery_snapshot_encode_capacity_overflow",
                                render_bvh.nodes.len(),
                                render_bvh.leaves.len(),
                            );
                        }
                    }
                } else if std::env::var_os("R4D_EDIT_RENDER_SYNC_DIAG").is_some() {
                    eprintln!(
                        "[edit-sync-voxel-delta] scene_bounds={:?}->{:?} result=ok applied_deltas={} frame_root={}",
                        scene_bounds.min,
                        scene_bounds.max,
                        applied_deltas,
                        self.voxel_frame_data.region_bvh_root_index
                    );
                    for delta in deltas.iter().take(8) {
                        let key = delta.key;
                        let world_payload = self.world_tree.chunk_payload(key).map(|(p, _)| p);
                        let cache_payload = self
                            .render_region_cache
                            .as_ref()
                            .and_then(|cache| cache.chunk_payload(key).map(|(p, _)| p));
                        let bvh_payloads = self
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
                            "[edit-sync-render] key={:?} root={:?}->{:?} writes(nodes={},leaves={},freed_nodes={},freed_leaves={}) world={} cache={} bvh={} frame={}",
                            key,
                            delta.expected_root,
                            delta.new_root,
                            delta.node_writes.len(),
                            delta.leaf_writes.len(),
                            delta.freed_node_ids.len(),
                            delta.freed_leaf_ids.len(),
                            Self::summarize_chunk_payload_compact(world_payload.as_ref()),
                            Self::summarize_chunk_payload_compact(cache_payload.as_ref()),
                            Self::summarize_chunk_payload_list_compact(&bvh_payloads),
                            Self::summarize_chunk_payload_list_compact(&frame_payloads),
                        );
                    }
                }
            }
        }

        &self.voxel_frame_data
    }

    /// Prime voxel frame metadata around the current spawn/camera position.
    pub fn preload_spawn_chunks(&mut self, spawn_pos: [f32; 4], max_trace_distance: f32, resolver: &MaterialResolver) {
        let start = Instant::now();
        let _ = self.build_voxel_frame_data(spawn_pos, [0.0, 0.0, 1.0, 0.0], max_trace_distance, resolver);
        eprintln!(
            "Preloaded render-tree voxel metadata in {:.2} ms",
            start.elapsed().as_secs_f64() * 1000.0
        );
    }
}
