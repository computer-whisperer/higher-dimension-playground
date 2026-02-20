use super::*;
use polychora::shared::chunk_payload::ChunkPayload;
use polychora::shared::render_tree::{RenderBvh, RenderBvhNodeKind, RenderLeafKind};
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

    fn dot4(a: [f32; 4], b: [f32; 4]) -> f32 {
        a[0] * b[0] + a[1] * b[1] + a[2] * b[2] + a[3] * b[3]
    }

    fn normalize4_with_fallback(v: [f32; 4], fallback: [f32; 4]) -> [f32; 4] {
        let len_sq = Self::dot4(v, v);
        if len_sq > 1e-12 {
            let inv_len = len_sq.sqrt().recip();
            [
                v[0] * inv_len,
                v[1] * inv_len,
                v[2] * inv_len,
                v[3] * inv_len,
            ]
        } else {
            fallback
        }
    }

    fn rebuild_voxel_frame_data_from_render_bvh(&mut self, render_bvh: &RenderBvh) {
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
            self.voxel_frame_data.chunk_headers.clear();
            self.voxel_frame_data.occupancy_words.clear();
            self.voxel_frame_data.material_words.clear();
            self.voxel_frame_data.macro_words.clear();
            self.voxel_frame_data.region_bvh_nodes.clear();
            self.voxel_frame_data.leaf_headers.clear();
            self.voxel_frame_data.leaf_chunk_entries.clear();
            self.voxel_visibility_generation = self.voxel_visibility_generation.wrapping_add(1);
            self.voxel_frame_data.metadata_generation = self.voxel_visibility_generation;
            return;
        }

        let mut dense_chunk_headers = Vec::<GpuVoxelChunkHeader>::new();
        let mut occupancy_words = Vec::<u32>::new();
        let mut material_words = Vec::<u32>::new();
        let mut macro_words = Vec::<u32>::new();
        let mut leaf_headers = Vec::<GpuVoxelLeafHeader>::with_capacity(render_bvh.leaves.len());
        let mut leaf_chunk_entries = Vec::<u32>::new();
        let mut region_bvh_nodes =
            Vec::<GpuVoxelChunkBvhNode>::with_capacity(render_bvh.nodes.len());

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
                                                if let Some(payload) = chunk_array
                                                    .chunk_palette
                                                    .get(palette_index as usize)
                                                {
                                                    encoded = match payload {
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
                                                            if dense_chunk_headers.len()
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
                                                                    chunk_index.saturating_add(1)
                                                                }
                                                            }
                                                        }
                                                    };
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
            let (left_child, right_child, leaf_index, flags) = match node.kind {
                RenderBvhNodeKind::Internal { left, right } => (left, right, u32::MAX, 0),
                RenderBvhNodeKind::Leaf { leaf_index } => (
                    higher_dimension_playground::render::VTE_REGION_BVH_INVALID_NODE,
                    higher_dimension_playground::render::VTE_REGION_BVH_INVALID_NODE,
                    leaf_index,
                    higher_dimension_playground::render::VTE_REGION_BVH_NODE_FLAG_LEAF,
                ),
            };
            region_bvh_nodes.push(GpuVoxelChunkBvhNode {
                min_chunk_coord: node.bounds.min,
                max_chunk_coord: node.bounds.max,
                left_child,
                right_child,
                leaf_index,
                flags,
            });
        }

        self.voxel_visibility_generation = self.voxel_visibility_generation.wrapping_add(1);
        self.voxel_frame_data.metadata_generation = self.voxel_visibility_generation;
        self.voxel_frame_data.chunk_headers = dense_chunk_headers;
        self.voxel_frame_data.occupancy_words = occupancy_words;
        self.voxel_frame_data.material_words = material_words;
        self.voxel_frame_data.macro_words = macro_words;
        self.voxel_frame_data.region_bvh_nodes = region_bvh_nodes;
        self.voxel_frame_data.leaf_headers = leaf_headers;
        self.voxel_frame_data.leaf_chunk_entries = leaf_chunk_entries;
    }

    /// Build the voxel-native frame payload for VTE.
    pub fn build_voxel_frame_data(
        &mut self,
        cam_pos: [f32; 4],
        cam_forward: [f32; 4],
        max_trace_distance: f32,
    ) -> &VoxelFrameData {
        let active_distance = max_trace_distance.max(VOXEL_NEAR_ACTIVE_DISTANCE);
        let chunk_radius = (active_distance / CHUNK_SIZE as f32).ceil() as i32 + 1;
        let cam_chunk = Self::camera_chunk_key(cam_pos);
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

        let cam_voxel = [
            cam_pos[0].floor() as i32,
            cam_pos[1].floor() as i32,
            cam_pos[2].floor() as i32,
            cam_pos[3].floor() as i32,
        ];
        let view_forward = Self::normalize4_with_fallback(cam_forward, [0.0, 0.0, 1.0, 0.0]);
        const VIEW_BIN_SCALE: f32 = 32.0;
        let cam_key = [
            cam_chunk[0],
            cam_chunk[1],
            cam_chunk[2],
            cam_chunk[3],
            cam_voxel[0],
            cam_voxel[1],
            cam_voxel[2],
            cam_voxel[3],
            (view_forward[0] * VIEW_BIN_SCALE).round() as i32,
            (view_forward[1] * VIEW_BIN_SCALE).round() as i32,
            (view_forward[2] * VIEW_BIN_SCALE).round() as i32,
            (view_forward[3] * VIEW_BIN_SCALE).round() as i32,
        ];
        let visibility_cache_valid = self.voxel_cached_visibility_camera_chunk == Some(cam_key)
            && self.voxel_cached_visibility_world_tree_revision == self.world_tree_revision;
        if !visibility_cache_valid {
            let render_bvh = self.render_bvh_cache.clone();
            if let Some(render_bvh) = render_bvh.as_ref() {
                self.rebuild_voxel_frame_data_from_render_bvh(render_bvh);
            } else {
                self.voxel_visibility_generation = self.voxel_visibility_generation.wrapping_add(1);
                self.voxel_frame_data.metadata_generation = self.voxel_visibility_generation;
                self.voxel_frame_data.chunk_headers.clear();
                self.voxel_frame_data.occupancy_words.clear();
                self.voxel_frame_data.material_words.clear();
                self.voxel_frame_data.macro_words.clear();
                self.voxel_frame_data.region_bvh_nodes.clear();
                self.voxel_frame_data.leaf_headers.clear();
                self.voxel_frame_data.leaf_chunk_entries.clear();
            }
            self.voxel_cached_visibility_camera_chunk = Some(cam_key);
            self.voxel_cached_visibility_world_tree_revision = self.world_tree_revision;
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
