use super::*;
use polychora::content_registry::MaterialResolver;
use polychora::shared::chunk_payload::ChunkPayload;
use polychora::shared::render_tree::{RenderBvhNodeKind, RenderLeaf, RenderLeafKind};
use polychora::shared::spatial::{fixed_from_lattice, Aabb4i, ChunkCoord};
use polychora::shared::voxel::BlockData;

/// Convert a `[ChunkCoord; 4]` to `[i32; 4]` for GPU boundary types.
#[inline]
pub(super) fn chunk_coord_to_i32(c: [ChunkCoord; 4]) -> [i32; 4] {
    [
        c[0].to_num::<i32>(),
        c[1].to_num::<i32>(),
        c[2].to_num::<i32>(),
        c[3].to_num::<i32>(),
    ]
}

/// Compute world-space AABB for a leaf's bounds at a given scale.
///
/// Bounds are already world-space half-open `[min, max)`.
/// Convert fixed-point to f32 for GPU consumption.
pub(super) fn leaf_world_bounds(
    bounds: &polychora::shared::spatial::Aabb4,
    _scale_exp: i8,
) -> ([f32; 4], [f32; 4]) {
    (
        bounds.min.map(|v| v.to_num::<f32>()),
        bounds.max.map(|v| v.to_num::<f32>()),
    )
}

/// Pack a scale_exp (i8) into the upper 8 bits (24-31) of a u32.
/// Used for the leaf header's `uniformOrientation` field (traversal grid scale).
#[inline]
pub(super) fn pack_scale_exp_into_orientation(orientation: u32, scale_exp: i8) -> u32 {
    (orientation & 0x00FFFFFFu32) | ((scale_exp as u8 as u32) << 24)
}

/// Pack tesseract orientation (9 bits, 0-383) and scale_exp (6 bits, signed)
/// into a 15-bit value for per-voxel and uniform chunk entry storage.
///
/// Layout: bits 0-8 = orientation, bits 9-14 = scale_exp (6-bit signed, range [-32, +31]).
/// Bit 15 is always 0 so the value fits in 15 bits (required for uniform chunk entries
/// where bit 15 of the u16 slot shares the UNIFORM_FLAG at bit 31 of the u32).
#[inline]
pub(super) fn pack_orientation_scale_u16(orientation: u16, scale_exp: i8) -> u16 {
    let ori = orientation & 0x1FF; // 9 bits
    let scale_bits = (scale_exp as u8 as u16) & 0x3F; // 6 bits
    ori | (scale_bits << 9)
}

pub(super) fn pack_dense_materials_words(
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

    // Pre-resolve the block palette so the inner loop uses array lookups
    // instead of HashMap lookups (palette is typically ~10-20 entries vs
    // 4096 voxels).
    let resolved_palette: Vec<(u16, u16)> = block_palette
        .iter()
        .map(|block| {
            let mat = resolver.resolve_block(block.namespace, block.block_type);
            let orient = pack_orientation_scale_u16(block.orientation.0, block.scale_exp);
            (mat, orient)
        })
        .collect();

    let mut solid_count = 0u32;
    let mut solid_local_min = [i32::MAX; 4];
    let mut solid_local_max = [i32::MIN; 4];

    for (voxel_idx, &palette_idx) in dense_palette_indices.iter().enumerate() {
        let (mat_u16, orient_scale_u16) = resolved_palette
            .get(palette_idx as usize)
            .copied()
            .unwrap_or((0, 0));

        let mat_word_idx = voxel_idx / 2;
        let mat_shift = ((voxel_idx & 1) * 16) as u32;
        material_words[mat_word_idx] &= !(0xFFFFu32 << mat_shift);
        material_words[mat_word_idx] |= u32::from(mat_u16) << mat_shift;

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
    pub(super) fn encode_bvh_node(
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
    pub(super) fn compute_all_delta_node_world_bounds(
        node_writes: &[(u32, polychora::shared::render_tree::RenderBvhNode)],
        gpu_nodes: &[GpuVoxelChunkBvhNode],
        leaf_headers: &[GpuVoxelLeafHeader],
        leaf_world_bounds_map: &std::collections::HashMap<u32, ([f32; 4], [f32; 4])>,
    ) -> std::collections::HashMap<u32, ([f32; 4], [f32; 4])> {
        // Index delta nodes by their slot ID for O(1) lookup during recursion.
        let delta_by_id: std::collections::HashMap<
            u32,
            &polychora::shared::render_tree::RenderBvhNode,
        > = node_writes.iter().map(|(id, node)| (*id, node)).collect();
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
        delta_by_id: &std::collections::HashMap<
            u32,
            &polychora::shared::render_tree::RenderBvhNode,
        >,
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
            return gpu_nodes
                .get(node_id as usize)
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
                    left,
                    delta_by_id,
                    gpu_nodes,
                    leaf_headers,
                    leaf_world_bounds_map,
                    computed,
                );
                let (rmin, rmax) = Self::resolve_node_world_bounds(
                    right,
                    delta_by_id,
                    gpu_nodes,
                    leaf_headers,
                    leaf_world_bounds_map,
                    computed,
                );
                (
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
                )
            }
        };
        computed.insert(node_id, bounds);
        bounds
    }

    pub(super) fn append_dense_payload_encoded(
        voxel_frame_data: &mut VoxelFrameData,
        dense_payload_cache: &mut std::collections::HashMap<DensePayloadCacheKey, u32>,
        payload: &ChunkPayload,
        block_palette: &[BlockData],
        resolver: &MaterialResolver,
    ) -> Result<u32, String> {
        let cache_key = DensePayloadCacheKey::new(payload, block_palette);
        if let Some(&encoded) = dense_payload_cache.get(&cache_key) {
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
        let (_solid_count, is_full, solid_local_min, solid_local_max) = pack_dense_materials_words(
            &dense_palette_indices,
            block_palette,
            &mut occ,
            &mut mat,
            &mut ori,
            &mut mac,
            resolver,
        );
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
        dense_payload_cache.insert(cache_key, encoded);
        Ok(encoded)
    }

    pub(super) fn encode_leaf_chunk_entries(
        voxel_frame_data: &mut VoxelFrameData,
        dense_payload_cache: &mut std::collections::HashMap<DensePayloadCacheKey, u32>,
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
        let (src_lattice_min, _src_lattice_max) =
            chunk_array.bounds.to_chunk_lattice_bounds(scale_exp);

        let mut encoded = Vec::<u32>::with_capacity(
            leaf.bounds
                .chunk_cell_count_at_scale(scale_exp)
                .unwrap_or(0),
        );
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
                            lat,
                            chunk_coord,
                            chunk_array.bounds,
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
                            ChunkPayload::Empty | ChunkPayload::Virgin => {
                                higher_dimension_playground::render::VTE_LEAF_CHUNK_ENTRY_EMPTY
                            }
                            ChunkPayload::Uniform(idx) => {
                                let block = chunk_array
                                    .block_palette
                                    .get(*idx as usize)
                                    .cloned()
                                    .unwrap_or(BlockData::AIR);
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
}
