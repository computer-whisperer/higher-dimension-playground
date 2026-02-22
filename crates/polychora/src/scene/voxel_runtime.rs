use super::*;
use polychora::materials::block_to_material_appearance;
use polychora::shared::chunk_payload::{ChunkPayload, ResolvedChunkPayload};
use polychora::shared::voxel::BlockData;
use polychora::shared::render_tree::{
    RenderBvh, RenderBvhChunkMutationDelta, RenderBvhNodeKind, RenderLeaf, RenderLeafKind,
};

use std::time::Instant;

fn pack_dense_materials_words(
    dense_palette_indices: &[u16],
    block_palette: &[BlockData],
    occupancy_words: &mut [u32; OCCUPANCY_WORDS_PER_CHUNK],
    material_words: &mut [u32; MATERIAL_WORDS_PER_CHUNK],
    macro_words: &mut [u32; MACRO_WORDS_PER_CHUNK],
) -> (u32, bool, [i32; 4], [i32; 4]) {
    debug_assert_eq!(dense_palette_indices.len(), CHUNK_VOLUME);

    occupancy_words.fill(0);
    material_words.fill(0);
    macro_words.fill(0);

    let mut solid_count = 0u32;
    let mut solid_local_min = [i32::MAX; 4];
    let mut solid_local_max = [i32::MIN; 4];

    for (voxel_idx, &palette_idx) in dense_palette_indices.iter().enumerate() {
        let block = block_palette
            .get(palette_idx as usize)
            .cloned()
            .unwrap_or(BlockData::AIR);
        let mat_u8 = block_to_material_appearance(block.namespace, block.block_type);

        let mat_word_idx = voxel_idx / 4;
        let mat_shift = ((voxel_idx & 3) * 8) as u32;
        material_words[mat_word_idx] &= !(0xFFu32 << mat_shift);
        material_words[mat_word_idx] |= u32::from(mat_u8) << mat_shift;

        if mat_u8 == 0 {
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
            min_chunk_coord: node.bounds.min,
            max_chunk_coord: node.bounds.max,
            left_child,
            right_child,
            leaf_index,
            flags,
        }
    }

    fn append_dense_payload_encoded(
        voxel_frame_data: &mut VoxelFrameData,
        dense_payload_cache: &mut std::collections::HashMap<ChunkPayload, u32>,
        payload: &ChunkPayload,
        block_palette: &[BlockData],
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
        let mut mac = [0u32; MACRO_WORDS_PER_CHUNK];
        let (_solid_count, is_full, solid_local_min, solid_local_max) =
            pack_dense_materials_words(&dense_palette_indices, block_palette, &mut occ, &mut mat, &mut mac);
        let chunk_index = voxel_frame_data.chunk_headers.len() as u32;
        let occ_offset = voxel_frame_data.occupancy_words.len() as u32;
        let mat_offset = voxel_frame_data.material_words.len() as u32;
        let mac_offset = voxel_frame_data.macro_words.len() as u32;
        voxel_frame_data.occupancy_words.extend_from_slice(&occ);
        voxel_frame_data.material_words.extend_from_slice(&mat);
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
            _padding: [0; 2],
        });
        let encoded = chunk_index.saturating_add(1);
        dense_payload_cache.insert(payload.clone(), encoded);
        Ok(encoded)
    }

    fn append_leaf_to_voxel_frame_data(
        voxel_frame_data: &mut VoxelFrameData,
        dense_payload_cache: &mut std::collections::HashMap<ChunkPayload, u32>,
        leaf: &RenderLeaf,
    ) -> Result<(), String> {
        match &leaf.kind {
            RenderLeafKind::Uniform(block) => {
                let mat = block_to_material_appearance(block.namespace, block.block_type);
                voxel_frame_data.leaf_headers.push(GpuVoxelLeafHeader {
                    min_chunk_coord: leaf.bounds.min,
                    max_chunk_coord: leaf.bounds.max,
                    leaf_kind: higher_dimension_playground::render::VTE_LEAF_KIND_UNIFORM,
                    uniform_material: u32::from(mat),
                    chunk_entry_offset: 0,
                    _padding: 0,
                });
            }
            RenderLeafKind::VoxelChunkArray(chunk_array) => {
                let Some(leaf_extents) = leaf.bounds.chunk_extents() else {
                    voxel_frame_data.leaf_headers.push(GpuVoxelLeafHeader {
                        min_chunk_coord: leaf.bounds.min,
                        max_chunk_coord: leaf.bounds.max,
                        leaf_kind:
                            higher_dimension_playground::render::VTE_LEAF_KIND_VOXEL_CHUNK_ARRAY,
                        uniform_material: 0,
                        chunk_entry_offset: 0,
                        _padding: 0,
                    });
                    return Ok(());
                };
                let entry_offset = voxel_frame_data.leaf_chunk_entries.len() as u32;
                let src_indices = chunk_array
                    .decode_dense_indices()
                    .map_err(|error| format!("decode chunk-array leaf failed: {error:?}"))?;
                let src_dims = chunk_array
                    .bounds
                    .chunk_extents()
                    .ok_or_else(|| "chunk-array source extents missing".to_string())?;

                for w in 0..leaf_extents[3] {
                    for z in 0..leaf_extents[2] {
                        for y in 0..leaf_extents[1] {
                            for x in 0..leaf_extents[0] {
                                let chunk_coord = [
                                    leaf.bounds.min[0] + x as i32,
                                    leaf.bounds.min[1] + y as i32,
                                    leaf.bounds.min[2] + z as i32,
                                    leaf.bounds.min[3] + w as i32,
                                ];
                                let lx = (chunk_coord[0] - chunk_array.bounds.min[0]) as usize;
                                let ly = (chunk_coord[1] - chunk_array.bounds.min[1]) as usize;
                                let lz = (chunk_coord[2] - chunk_array.bounds.min[2]) as usize;
                                let lw = (chunk_coord[3] - chunk_array.bounds.min[3]) as usize;
                                let linear =
                                    lx + src_dims[0] * (ly + src_dims[1] * (lz + src_dims[2] * lw));
                                let palette_index =
                                    src_indices.get(linear).copied().ok_or_else(|| {
                                        "chunk-array index out of bounds".to_string()
                                    })? as usize;
                                let payload =
                                    chunk_array.chunk_palette.get(palette_index).ok_or_else(
                                        || "chunk-array palette out of bounds".to_string(),
                                    )?;
                                let encoded = match payload {
                                    ChunkPayload::Empty => {
                                        higher_dimension_playground::render::VTE_LEAF_CHUNK_ENTRY_EMPTY
                                    }
                                    ChunkPayload::Uniform(idx) => {
                                        let block = chunk_array.block_palette.get(*idx as usize).cloned().unwrap_or(BlockData::AIR);
                                        let mat = block_to_material_appearance(block.namespace, block.block_type);
                                        if mat == 0 {
                                            higher_dimension_playground::render::VTE_LEAF_CHUNK_ENTRY_EMPTY
                                        } else {
                                            higher_dimension_playground::render::VTE_LEAF_CHUNK_ENTRY_UNIFORM_FLAG
                                                | u32::from(mat)
                                        }
                                    }
                                    dense_payload => {
                                        Self::append_dense_payload_encoded(
                                            voxel_frame_data,
                                            dense_payload_cache,
                                            dense_payload,
                                            &chunk_array.block_palette,
                                        )?
                                    }
                                };
                                voxel_frame_data.leaf_chunk_entries.push(encoded);
                            }
                        }
                    }
                }

                voxel_frame_data.leaf_headers.push(GpuVoxelLeafHeader {
                    min_chunk_coord: leaf.bounds.min,
                    max_chunk_coord: leaf.bounds.max,
                    leaf_kind: higher_dimension_playground::render::VTE_LEAF_KIND_VOXEL_CHUNK_ARRAY,
                    uniform_material: 0,
                    chunk_entry_offset: entry_offset,
                    _padding: 0,
                });
            }
        }
        Ok(())
    }

    fn invalid_leaf_header() -> GpuVoxelLeafHeader {
        GpuVoxelLeafHeader {
            min_chunk_coord: [0, 0, 0, 0],
            max_chunk_coord: [-1, -1, -1, -1],
            leaf_kind: higher_dimension_playground::render::VTE_LEAF_KIND_UNIFORM,
            uniform_material: 0,
            chunk_entry_offset: 0,
            _padding: 0,
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
    ) -> Result<Vec<u32>, String> {
        let RenderLeafKind::VoxelChunkArray(chunk_array) = &leaf.kind else {
            return Ok(Vec::new());
        };
        let Some(leaf_extents) = leaf.bounds.chunk_extents() else {
            return Ok(Vec::new());
        };
        let src_indices = chunk_array
            .decode_dense_indices()
            .map_err(|error| format!("decode chunk-array leaf failed: {error:?}"))?;
        let src_dims = chunk_array
            .bounds
            .chunk_extents()
            .ok_or_else(|| "chunk-array source extents missing".to_string())?;

        let mut encoded = Vec::<u32>::with_capacity(leaf.bounds.chunk_cell_count().unwrap_or(0));
        for w in 0..leaf_extents[3] {
            for z in 0..leaf_extents[2] {
                for y in 0..leaf_extents[1] {
                    for x in 0..leaf_extents[0] {
                        let chunk_coord = [
                            leaf.bounds.min[0] + x as i32,
                            leaf.bounds.min[1] + y as i32,
                            leaf.bounds.min[2] + z as i32,
                            leaf.bounds.min[3] + w as i32,
                        ];
                        let lx = (chunk_coord[0] - chunk_array.bounds.min[0]) as usize;
                        let ly = (chunk_coord[1] - chunk_array.bounds.min[1]) as usize;
                        let lz = (chunk_coord[2] - chunk_array.bounds.min[2]) as usize;
                        let lw = (chunk_coord[3] - chunk_array.bounds.min[3]) as usize;
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
                                let mat = block_to_material_appearance(block.namespace, block.block_type);
                                if mat == 0 {
                                    higher_dimension_playground::render::VTE_LEAF_CHUNK_ENTRY_EMPTY
                                } else {
                                    higher_dimension_playground::render::VTE_LEAF_CHUNK_ENTRY_UNIFORM_FLAG
                                        | u32::from(mat)
                                }
                            }
                            dense_payload => Self::append_dense_payload_encoded(
                                voxel_frame_data,
                                dense_payload_cache,
                                dense_payload,
                                &chunk_array.block_palette,
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
            let bounds = Aabb4i::new(header.min_chunk_coord, header.max_chunk_coord);
            let Some(cell_count) = bounds.chunk_cell_count() else {
                continue;
            };
            if cell_count == 0 {
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
            for (node_index, src_node) in &delta.node_writes {
                let node_index = *node_index as usize;
                if self.voxel_frame_data.region_bvh_nodes.len() <= node_index {
                    self.voxel_frame_data
                        .region_bvh_nodes
                        .resize(node_index + 1, GpuVoxelChunkBvhNode::empty());
                }
                self.voxel_frame_data.region_bvh_nodes[node_index] =
                    Self::encode_bvh_node(src_node);
                Self::mark_dirty_index(&mut dirty.region_bvh_nodes, node_index);
            }

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
                        let mat = block_to_material_appearance(block.namespace, block.block_type);
                        self.voxel_frame_data.leaf_headers[leaf_index] = GpuVoxelLeafHeader {
                            min_chunk_coord: leaf.bounds.min,
                            max_chunk_coord: leaf.bounds.max,
                            leaf_kind: higher_dimension_playground::render::VTE_LEAF_KIND_UNIFORM,
                            uniform_material: u32::from(mat),
                            chunk_entry_offset: 0,
                            _padding: 0,
                        };
                        Self::mark_dirty_index(&mut dirty.leaf_headers, leaf_index);
                    }
                    RenderLeafKind::VoxelChunkArray(_) => {
                        let entries = Self::encode_leaf_chunk_entries(
                            &mut self.voxel_frame_data,
                            &mut self.voxel_dense_payload_encoded_cache,
                            leaf,
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
                            min_chunk_coord: leaf.bounds.min,
                            max_chunk_coord: leaf.bounds.max,
                            leaf_kind:
                                higher_dimension_playground::render::VTE_LEAF_KIND_VOXEL_CHUNK_ARRAY,
                            uniform_material: 0,
                            chunk_entry_offset: span.start as u32,
                            _padding: 0,
                        };
                        Self::mark_dirty_index(&mut dirty.leaf_headers, leaf_index);
                    }
                }
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
    ) -> Option<VoxelFrameDataBuffers> {
        let mut dense_chunk_headers = Vec::<GpuVoxelChunkHeader>::new();
        let mut occupancy_words = Vec::<u32>::new();
        let mut material_words = Vec::<u32>::new();
        let mut macro_words = Vec::<u32>::new();
        let mut leaf_headers = Vec::<GpuVoxelLeafHeader>::with_capacity(render_bvh.leaves.len());
        let mut leaf_chunk_entries = Vec::<u32>::new();
        let mut region_bvh_nodes =
            Vec::<GpuVoxelChunkBvhNode>::with_capacity(render_bvh.nodes.len());
        let mut dense_payload_encoded_cache = std::collections::HashMap::<ChunkPayload, u32>::new();

        for leaf in &render_bvh.leaves {
            match &leaf.kind {
                RenderLeafKind::Uniform(block) => {
                    let mat = block_to_material_appearance(block.namespace, block.block_type);
                    leaf_headers.push(GpuVoxelLeafHeader {
                        min_chunk_coord: leaf.bounds.min,
                        max_chunk_coord: leaf.bounds.max,
                        leaf_kind: higher_dimension_playground::render::VTE_LEAF_KIND_UNIFORM,
                        uniform_material: u32::from(mat),
                        chunk_entry_offset: 0,
                        _padding: 0,
                    });
                }
                RenderLeafKind::VoxelChunkArray(chunk_array) => {
                    let Some(leaf_extents) = leaf.bounds.chunk_extents() else {
                        leaf_headers.push(GpuVoxelLeafHeader {
                            min_chunk_coord: leaf.bounds.min,
                            max_chunk_coord: leaf.bounds.max,
                            leaf_kind: higher_dimension_playground::render::VTE_LEAF_KIND_VOXEL_CHUNK_ARRAY,
                            uniform_material: 0,
                            chunk_entry_offset: 0,
                            _padding: 0,
                        });
                        continue;
                    };
                    let leaf_cell_count = leaf.bounds.chunk_cell_count().unwrap_or(0);
                    let entry_offset = leaf_chunk_entries.len() as u32;
                    leaf_chunk_entries.reserve(leaf_cell_count);

                    let src_indices = chunk_array.decode_dense_indices().ok();
                    let src_dims = chunk_array.bounds.chunk_extents();
                    let mut palette_encoded_cache =
                        vec![None::<u32>; chunk_array.chunk_palette.len()];

                    for w in 0..leaf_extents[3] {
                        for z in 0..leaf_extents[2] {
                            for y in 0..leaf_extents[1] {
                                for x in 0..leaf_extents[0] {
                                    let chunk_coord = [
                                        leaf.bounds.min[0] + x as i32,
                                        leaf.bounds.min[1] + y as i32,
                                        leaf.bounds.min[2] + z as i32,
                                        leaf.bounds.min[3] + w as i32,
                                    ];
                                    let mut encoded = higher_dimension_playground::render::VTE_LEAF_CHUNK_ENTRY_EMPTY;
                                    if let (Some(indices), Some(src_dims)) =
                                        (src_indices.as_ref(), src_dims)
                                    {
                                        if chunk_coord[0] >= chunk_array.bounds.min[0]
                                            && chunk_coord[0] <= chunk_array.bounds.max[0]
                                            && chunk_coord[1] >= chunk_array.bounds.min[1]
                                            && chunk_coord[1] <= chunk_array.bounds.max[1]
                                            && chunk_coord[2] >= chunk_array.bounds.min[2]
                                            && chunk_coord[2] <= chunk_array.bounds.max[2]
                                            && chunk_coord[3] >= chunk_array.bounds.min[3]
                                            && chunk_coord[3] <= chunk_array.bounds.max[3]
                                        {
                                            let lx = (chunk_coord[0] - chunk_array.bounds.min[0])
                                                as usize;
                                            let ly = (chunk_coord[1] - chunk_array.bounds.min[1])
                                                as usize;
                                            let lz = (chunk_coord[2] - chunk_array.bounds.min[2])
                                                as usize;
                                            let lw = (chunk_coord[3] - chunk_array.bounds.min[3])
                                                as usize;
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
                                                            let mat = block_to_material_appearance(block.namespace, block.block_type);
                                                            if mat == 0 {
                                                                higher_dimension_playground::render::VTE_LEAF_CHUNK_ENTRY_EMPTY
                                                            } else {
                                                                higher_dimension_playground::render::VTE_LEAF_CHUNK_ENTRY_UNIFORM_FLAG
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
                                                                    let mut mac = [0u32;
                                                                        MACRO_WORDS_PER_CHUNK];
                                                                    let (_solid_count, is_full, solid_local_min, solid_local_max) =
                                                                        pack_dense_materials_words(
                                                                            &dense_materials,
                                                                            &chunk_array.block_palette,
                                                                            &mut occ,
                                                                            &mut mat,
                                                                            &mut mac,
                                                                        );
                                                                    let chunk_index =
                                                                        dense_chunk_headers.len() as u32;
                                                                    let occ_offset =
                                                                        occupancy_words.len() as u32;
                                                                    let mat_offset =
                                                                        material_words.len() as u32;
                                                                    let mac_offset =
                                                                        macro_words.len() as u32;
                                                                    occupancy_words
                                                                        .extend_from_slice(&occ);
                                                                    material_words
                                                                        .extend_from_slice(&mat);
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
                                                                            _padding: [0; 2],
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
                        min_chunk_coord: leaf.bounds.min,
                        max_chunk_coord: leaf.bounds.max,
                        leaf_kind:
                            higher_dimension_playground::render::VTE_LEAF_KIND_VOXEL_CHUNK_ARRAY,
                        uniform_material: 0,
                        chunk_entry_offset: entry_offset,
                        _padding: 0,
                    });
                }
            }
        }

        for node in &render_bvh.nodes {
            region_bvh_nodes.push(Self::encode_bvh_node(node));
        }

        Some(VoxelFrameDataBuffers {
            region_bvh_root_index: render_bvh
                .root
                .unwrap_or(higher_dimension_playground::render::VTE_REGION_BVH_INVALID_NODE),
            dense_payload_encoded_cache,
            chunk_headers: dense_chunk_headers,
            occupancy_words,
            material_words,
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
    ) -> &VoxelFrameData {
        const SCENE_RESIDENCY_RADIUS_MULTIPLIER: i32 = 2;
        const SCENE_RESIDENCY_EXTRA_CHUNKS: i32 = 2;

        let active_distance = max_trace_distance.max(VOXEL_NEAR_ACTIVE_DISTANCE);
        let chunk_radius = (active_distance / CHUNK_SIZE as f32).ceil() as i32 + 1;
        let cam_chunk = Self::camera_chunk_key(cam_pos);
        let view_bounds = Aabb4i::new(
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
        let resident_radius = chunk_radius
            .saturating_mul(SCENE_RESIDENCY_RADIUS_MULTIPLIER)
            .saturating_add(SCENE_RESIDENCY_EXTRA_CHUNKS)
            .max(chunk_radius + 1);
        let desired_scene_bounds = Aabb4i::new(
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
                        Self::build_voxel_frame_buffers_from_render_bvh(render_bvh)
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
                        .apply_render_bvh_mutation_deltas_to_voxel_frame_data(&deltas, scene_bounds)
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
                            Self::build_voxel_frame_buffers_from_render_bvh(render_bvh)
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
                        let world_payload = self.world_tree.chunk_payload(key);
                        let cache_payload = self
                            .render_region_cache
                            .as_ref()
                            .and_then(|cache| cache.chunk_payload(key));
                        let bvh_payloads = self
                            .render_bvh_cache
                            .as_ref()
                            .map(|bvh| render_tree::sample_chunk_payloads_from_bvh(bvh, key))
                            .unwrap_or_default();
                        let frame_payloads: Vec<ResolvedChunkPayload> = self
                            .debug_voxel_frame_chunk_payloads(key)
                            .into_iter()
                            .map(ResolvedChunkPayload::from_legacy_payload)
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
    pub fn preload_spawn_chunks(&mut self, spawn_pos: [f32; 4], max_trace_distance: f32) {
        let start = Instant::now();
        let _ = self.build_voxel_frame_data(spawn_pos, [0.0, 0.0, 1.0, 0.0], max_trace_distance);
        eprintln!(
            "Preloaded render-tree voxel metadata in {:.2} ms",
            start.elapsed().as_secs_f64() * 1000.0
        );
    }
}
