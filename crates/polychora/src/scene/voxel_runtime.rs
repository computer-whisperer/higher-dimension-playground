use super::*;
use polychora::shared::chunk_payload::ChunkPayload;
use polychora::shared::render_tree::{
    RenderBvh, RenderBvhChunkMutationDelta, RenderBvhNodeKind, RenderLeaf, RenderLeafKind,
};
use std::time::Instant;

fn pack_dense_materials_words(
    dense_materials: &[u16],
    occupancy_words: &mut [u32; OCCUPANCY_WORDS_PER_CHUNK],
    material_words: &mut [u32; MATERIAL_WORDS_PER_CHUNK],
    macro_words: &mut [u32; MACRO_WORDS_PER_CHUNK],
) -> (u32, bool, [i32; 4], [i32; 4]) {
    debug_assert_eq!(dense_materials.len(), CHUNK_VOLUME);

    occupancy_words.fill(0);
    material_words.fill(0);
    macro_words.fill(0);

    let mut solid_count = 0u32;
    let mut solid_local_min = [i32::MAX; 4];
    let mut solid_local_max = [i32::MIN; 4];

    for (voxel_idx, material) in dense_materials.iter().copied().enumerate() {
        let mat_word_idx = voxel_idx / 4;
        let mat_shift = ((voxel_idx & 3) * 8) as u32;
        let mat_u8 = u8::try_from(material).unwrap_or(u8::MAX);
        material_words[mat_word_idx] &= !(0xFFu32 << mat_shift);
        material_words[mat_word_idx] |= u32::from(mat_u8) << mat_shift;

        if material == 0 {
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
    ) -> Result<u32, String> {
        if let Some(&encoded) = dense_payload_cache.get(payload) {
            return Ok(encoded);
        }
        if voxel_frame_data.chunk_headers.len() >= VTE_MAX_DENSE_CHUNKS {
            return Err("dense chunk capacity exceeded".to_string());
        }
        let dense_materials = payload
            .dense_materials()
            .map_err(|error| format!("dense payload decode failed: {error}"))?;
        if dense_materials.len() != CHUNK_VOLUME {
            return Err(format!(
                "dense payload voxel count {} != {}",
                dense_materials.len(),
                CHUNK_VOLUME
            ));
        }
        let mut occ = [0u32; OCCUPANCY_WORDS_PER_CHUNK];
        let mut mat = [0u32; MATERIAL_WORDS_PER_CHUNK];
        let mut mac = [0u32; MACRO_WORDS_PER_CHUNK];
        let (_solid_count, is_full, solid_local_min, solid_local_max) =
            pack_dense_materials_words(&dense_materials, &mut occ, &mut mat, &mut mac);
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
        if voxel_frame_data.leaf_headers.len() >= VTE_REGION_LEAF_CAPACITY {
            return Err("leaf capacity exceeded".to_string());
        }

        match &leaf.kind {
            RenderLeafKind::Uniform(material) => {
                voxel_frame_data.leaf_headers.push(GpuVoxelLeafHeader {
                    min_chunk_coord: leaf.bounds.min,
                    max_chunk_coord: leaf.bounds.max,
                    leaf_kind: higher_dimension_playground::render::VTE_LEAF_KIND_UNIFORM,
                    uniform_material: u32::from(*material),
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
                let leaf_cell_count = leaf.bounds.chunk_cell_count().unwrap_or(0);
                if voxel_frame_data
                    .leaf_chunk_entries
                    .len()
                    .saturating_add(leaf_cell_count)
                    > VTE_REGION_LEAF_CHUNK_ENTRY_CAPACITY
                {
                    return Err("leaf chunk-entry capacity exceeded".to_string());
                }
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
                                    ChunkPayload::Empty | ChunkPayload::Uniform(0) => {
                                        higher_dimension_playground::render::VTE_LEAF_CHUNK_ENTRY_EMPTY
                                    }
                                    ChunkPayload::Uniform(material) => {
                                        higher_dimension_playground::render::VTE_LEAF_CHUNK_ENTRY_UNIFORM_FLAG
                                            | u32::from(*material)
                                    }
                                    dense_payload => {
                                        Self::append_dense_payload_encoded(
                                            voxel_frame_data,
                                            dense_payload_cache,
                                            dense_payload,
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
                        let linear = lx + src_dims[0] * (ly + src_dims[1] * (lz + src_dims[2] * lw));
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
                            ChunkPayload::Empty | ChunkPayload::Uniform(0) => {
                                higher_dimension_playground::render::VTE_LEAF_CHUNK_ENTRY_EMPTY
                            }
                            ChunkPayload::Uniform(material) => {
                                higher_dimension_playground::render::VTE_LEAF_CHUNK_ENTRY_UNIFORM_FLAG
                                    | u32::from(*material)
                            }
                            dense_payload => Self::append_dense_payload_encoded(
                                voxel_frame_data,
                                dense_payload_cache,
                                dense_payload,
                            )?,
                        };
                        encoded.push(encoded_entry);
                    }
                }
            }
        }
        Ok(encoded)
    }

    fn rebuild_leaf_entry_spans_from_headers(
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
        if used_len < VTE_REGION_LEAF_CHUNK_ENTRY_CAPACITY {
            free.push(used_len..VTE_REGION_LEAF_CHUNK_ENTRY_CAPACITY);
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
            return Err("leaf chunk-entry arena exhausted".to_string());
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
        render_bvh: &RenderBvh,
        deltas: &[RenderBvhChunkMutationDelta],
        bounds: Aabb4i,
    ) -> Result<(), String> {
        if !bounds.is_valid() {
            return Ok(());
        }
        let mut leaf_spans = Self::rebuild_leaf_entry_spans_from_headers(&self.voxel_frame_data);
        let mut free_spans = Self::build_free_spans_from_leaf_spans(
            &leaf_spans,
            self.voxel_frame_data.leaf_chunk_entries.len(),
        );

        for delta in deltas {
            for (node_index, src_node) in &delta.node_writes {
                let node_index = *node_index as usize;
                if node_index >= VTE_REGION_BVH_NODE_CAPACITY {
                    return Err(format!(
                        "node write index {} exceeds capacity {}",
                        node_index, VTE_REGION_BVH_NODE_CAPACITY
                    ));
                }
                if self.voxel_frame_data.region_bvh_nodes.len() <= node_index {
                    self.voxel_frame_data
                        .region_bvh_nodes
                        .resize(node_index + 1, GpuVoxelChunkBvhNode::empty());
                }
                self.voxel_frame_data.region_bvh_nodes[node_index] = Self::encode_bvh_node(src_node);
            }

            for node_index in &delta.freed_node_ids {
                let node_index = *node_index as usize;
                if node_index < self.voxel_frame_data.region_bvh_nodes.len() {
                    self.voxel_frame_data.region_bvh_nodes[node_index] = GpuVoxelChunkBvhNode::empty();
                }
            }

            for leaf_index in &delta.freed_leaf_ids {
                let leaf_index = *leaf_index as usize;
                if leaf_index < leaf_spans.len() {
                    if let Some(old_span) = leaf_spans[leaf_index].take() {
                        Self::merge_free_span(&mut free_spans, old_span);
                    }
                }
                if leaf_index < self.voxel_frame_data.leaf_headers.len() {
                    self.voxel_frame_data.leaf_headers[leaf_index] = Self::invalid_leaf_header();
                }
            }

            for (leaf_index, leaf) in &delta.leaf_writes {
                let leaf_index = *leaf_index as usize;
                if leaf_index >= VTE_REGION_LEAF_CAPACITY {
                    return Err(format!(
                        "leaf write index {} exceeds capacity {}",
                        leaf_index, VTE_REGION_LEAF_CAPACITY
                    ));
                }
                if self.voxel_frame_data.leaf_headers.len() <= leaf_index {
                    self.voxel_frame_data
                        .leaf_headers
                        .resize(leaf_index + 1, Self::invalid_leaf_header());
                }
                if leaf_spans.len() <= leaf_index {
                    leaf_spans.resize(leaf_index + 1, None);
                }
                if let Some(old_span) = leaf_spans[leaf_index].take() {
                    Self::merge_free_span(&mut free_spans, old_span);
                }

                match &leaf.kind {
                    RenderLeafKind::Uniform(material) => {
                        self.voxel_frame_data.leaf_headers[leaf_index] = GpuVoxelLeafHeader {
                            min_chunk_coord: leaf.bounds.min,
                            max_chunk_coord: leaf.bounds.max,
                            leaf_kind: higher_dimension_playground::render::VTE_LEAF_KIND_UNIFORM,
                            uniform_material: u32::from(*material),
                            chunk_entry_offset: 0,
                            _padding: 0,
                        };
                    }
                    RenderLeafKind::VoxelChunkArray(_) => {
                        let entries = Self::encode_leaf_chunk_entries(
                            &mut self.voxel_frame_data,
                            &mut self.voxel_dense_payload_encoded_cache,
                            leaf,
                        )?;
                        let span = Self::allocate_leaf_entry_span(
                            &mut free_spans,
                            &mut self.voxel_frame_data.leaf_chunk_entries,
                            entries.len(),
                        )?;
                        self.voxel_frame_data.leaf_chunk_entries[span.clone()]
                            .copy_from_slice(entries.as_slice());
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
                    }
                }
            }
        }

        self.voxel_visibility_generation = self.voxel_visibility_generation.wrapping_add(1);
        self.voxel_frame_data.metadata_generation = self.voxel_visibility_generation;
        self.voxel_frame_data.region_bvh_root_index = render_bvh
            .root
            .unwrap_or(higher_dimension_playground::render::VTE_REGION_BVH_INVALID_NODE);
        self.voxel_cached_visibility_bounds = Some(bounds);
        Ok(())
    }

    fn build_voxel_frame_buffers_from_render_bvh(
        render_bvh: &RenderBvh,
    ) -> Option<VoxelFrameDataBuffers> {
        if render_bvh.nodes.len() > VTE_REGION_BVH_NODE_CAPACITY
            || render_bvh.leaves.len() > VTE_REGION_LEAF_CAPACITY
        {
            eprintln!(
                "VTE render tree overflow: nodes {}>{}, leaves {}>{}; dropping frame tree.",
                render_bvh.nodes.len(),
                VTE_REGION_BVH_NODE_CAPACITY,
                render_bvh.leaves.len(),
                VTE_REGION_LEAF_CAPACITY
            );
            return None;
        }

        let mut dense_chunk_headers = Vec::<GpuVoxelChunkHeader>::new();
        let mut occupancy_words = Vec::<u32>::new();
        let mut material_words = Vec::<u32>::new();
        let mut macro_words = Vec::<u32>::new();
        let mut leaf_headers = Vec::<GpuVoxelLeafHeader>::with_capacity(render_bvh.leaves.len());
        let mut leaf_chunk_entries = Vec::<u32>::new();
        let mut region_bvh_nodes =
            Vec::<GpuVoxelChunkBvhNode>::with_capacity(render_bvh.nodes.len());
        let mut dense_payload_encoded_cache = std::collections::HashMap::<ChunkPayload, u32>::new();

        let mut overflowed = false;
        for leaf in &render_bvh.leaves {
            match &leaf.kind {
                RenderLeafKind::Uniform(material) => {
                    leaf_headers.push(GpuVoxelLeafHeader {
                        min_chunk_coord: leaf.bounds.min,
                        max_chunk_coord: leaf.bounds.max,
                        leaf_kind: higher_dimension_playground::render::VTE_LEAF_KIND_UNIFORM,
                        uniform_material: u32::from(*material),
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
                    if leaf_chunk_entries.len().saturating_add(leaf_cell_count)
                        > VTE_REGION_LEAF_CHUNK_ENTRY_CAPACITY
                    {
                        overflowed = true;
                        break;
                    }

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
                                                        ChunkPayload::Uniform(0) => {
                                                            higher_dimension_playground::render::VTE_LEAF_CHUNK_ENTRY_EMPTY
                                                        }
                                                        ChunkPayload::Uniform(material) => {
                                                            higher_dimension_playground::render::VTE_LEAF_CHUNK_ENTRY_UNIFORM_FLAG
                                                                | u32::from(*material)
                                                        }
                                                        _ => {
                                                            if let Some(&cached_dense_encoded) =
                                                                dense_payload_encoded_cache
                                                                    .get(payload)
                                                            {
                                                                cached_dense_encoded
                                                            } else if dense_chunk_headers.len()
                                                                >= VTE_MAX_DENSE_CHUNKS
                                                            {
                                                                overflowed = true;
                                                                higher_dimension_playground::render::VTE_LEAF_CHUNK_ENTRY_EMPTY
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
                                    if overflowed {
                                        break;
                                    }
                                }
                                if overflowed {
                                    break;
                                }
                            }
                            if overflowed {
                                break;
                            }
                        }
                        if overflowed {
                            break;
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
            if overflowed {
                break;
            }
        }

        if overflowed {
            eprintln!(
                "VTE voxel tree staging overflow; truncated dense chunks={}, leaf entries={}.",
                dense_chunk_headers.len(),
                leaf_chunk_entries.len()
            );
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
        self.voxel_dense_payload_encoded_cache.clear();
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
        } else {
            self.clear_voxel_frame_buffers();
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

        let visibility_cache_valid = self.voxel_cached_visibility_bounds == Some(scene_bounds)
            && !self.voxel_scene_bounds_has_pending_dirty_regions(scene_bounds);
        if !visibility_cache_valid {
            self.ensure_render_bvh_cache_for_bounds(scene_bounds);
            let (needs_rebuild, deltas) = self.take_pending_render_bvh_update_flags();

            if needs_rebuild {
                let buffers = self
                    .render_bvh_cache
                    .as_ref()
                    .and_then(Self::build_voxel_frame_buffers_from_render_bvh);
                self.apply_voxel_frame_buffers(scene_bounds, buffers);
            } else if !deltas.is_empty() {
                let mut apply_ok = false;
                if let Some(render_bvh) = self.render_bvh_cache.take() {
                    apply_ok = self
                        .apply_render_bvh_mutation_deltas_to_voxel_frame_data(
                            &render_bvh,
                            &deltas,
                            scene_bounds,
                        )
                        .is_ok();
                    self.render_bvh_cache = Some(render_bvh);
                }

                if !apply_ok {
                    let buffers = self
                        .render_bvh_cache
                        .as_ref()
                        .and_then(Self::build_voxel_frame_buffers_from_render_bvh);
                    self.apply_voxel_frame_buffers(scene_bounds, buffers);
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
