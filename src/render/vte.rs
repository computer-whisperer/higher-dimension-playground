use bytemuck::{Pod, Zeroable};

#[derive(Copy, Clone, Debug, Default, Pod, Zeroable)]
#[repr(C)]
pub struct GpuVoxelChunkHeader {
    pub occupancy_word_offset: u32,
    pub material_word_offset: u32,
    pub flags: u32,
    pub macro_word_offset: u32,
    pub solid_local_min: [i32; 4],
    pub solid_local_max: [i32; 4],
    pub _padding: [u32; 2],
}

impl GpuVoxelChunkHeader {
    pub const FLAG_FULL: u32 = 1 << 0;
}

#[derive(Copy, Clone, Debug, Default, Pod, Zeroable)]
#[repr(C)]
pub struct GpuVoxelLeafHeader {
    pub min_chunk_coord: [i32; 4],
    pub max_chunk_coord: [i32; 4],
    pub leaf_kind: u32,
    pub uniform_material: u32,
    pub chunk_entry_offset: u32,
    pub _padding: u32,
}

pub struct VoxelFrameInput<'a> {
    pub metadata_generation: u64,
    pub chunk_headers: &'a [GpuVoxelChunkHeader],
    pub occupancy_words: &'a [u32],
    pub material_words: &'a [u32],
    pub macro_words: &'a [u32],
    pub region_bvh_nodes: &'a [GpuVoxelChunkBvhNode],
    pub leaf_headers: &'a [GpuVoxelLeafHeader],
    pub leaf_chunk_entries: &'a [u32],
}

#[derive(Copy, Clone, Default, Pod, Zeroable)]
#[repr(C)]
pub(super) struct GpuVoxelFrameMeta {
    pub(super) chunk_count: u32,
    pub(super) leaf_count: u32,
    pub(super) occupancy_word_count: u32,
    pub(super) material_word_count: u32,
    pub(super) macro_word_count: u32,
    pub(super) max_trace_steps: u32,
    pub(super) max_trace_distance: f32,
    pub(super) region_bvh_node_count: u32,
    pub(super) _reserved0: u32,
    pub(super) leaf_chunk_entry_count: u32,
    pub(super) stage_b_mode: u32,
    pub(super) stage_b_slice_layer: u32,
    pub(super) stage_b_thick_half_width: u32,
    pub(super) debug_flags: u32,
    pub(super) visible_chunk_min_x: i32,
    pub(super) visible_chunk_min_y: i32,
    pub(super) visible_chunk_min_z: i32,
    pub(super) visible_chunk_min_w: i32,
    pub(super) visible_chunk_max_x: i32,
    pub(super) visible_chunk_max_y: i32,
    pub(super) visible_chunk_max_z: i32,
    pub(super) visible_chunk_max_w: i32,
    pub(super) highlight_flags: u32,
    pub(super) _highlight_padding: [u32; 3],
    pub(super) highlight_hit_voxel: [i32; 4],
    pub(super) highlight_place_voxel: [i32; 4],
}

#[derive(Copy, Clone, Debug, Pod, Zeroable)]
#[repr(C)]
pub struct GpuVoxelChunkBvhNode {
    pub min_chunk_coord: [i32; 4],
    pub max_chunk_coord: [i32; 4],
    pub left_child: u32,
    pub right_child: u32,
    pub leaf_index: u32,
    pub flags: u32,
}

impl GpuVoxelChunkBvhNode {
    pub fn empty() -> Self {
        Self {
            min_chunk_coord: [0; 4],
            max_chunk_coord: [0; 4],
            left_child: VTE_REGION_BVH_INVALID_NODE,
            right_child: VTE_REGION_BVH_INVALID_NODE,
            leaf_index: u32::MAX,
            flags: 0,
        }
    }
}

impl Default for GpuVoxelChunkBvhNode {
    fn default() -> Self {
        Self::empty()
    }
}

#[derive(Copy, Clone, Default)]
pub struct VteDebugCounters {
    pub candidate_chunks: u32,
    pub frustum_culled_chunks: u32,
    pub empty_chunks_skipped: u32,
    pub macro_cells_skipped: u32,
    pub chunk_steps: u64,
    pub voxel_steps: u64,
    pub primary_hits: u64,
    pub s_samples: u64,
    pub visible_set_hash_valid: bool,
    pub visible_set_hash: u32,
}

#[derive(Copy, Clone, Default)]
pub(super) struct VteCompareStats {
    pub(super) compared: u32,
    pub(super) matches: u32,
    pub(super) mismatches: u32,
    pub(super) hit_state_mismatches: u32,
    pub(super) chunk_material_mismatches: u32,
    pub(super) fast_miss_ref_hit: u32,
    pub(super) fast_hit_ref_miss: u32,
    pub(super) miss_reason_counts: [u32; 6],
    pub(super) zero_interval_flags: u32,
    pub(super) tie_stepped_flags: u32,
    pub(super) lookup_fallback_flags: u32,
    pub(super) entity_bvh_samples: u32,
    pub(super) entity_bvh_mismatches: u32,
    pub(super) entity_bvh_hit_state_mismatches: u32,
    pub(super) entity_bvh_material_mismatches: u32,
    pub(super) entity_bvh_distance_mismatches: u32,
    pub(super) entity_bvh_tetra_mismatches: u32,
    pub(super) entity_bvh_miss_linear_hit: u32,
    pub(super) entity_bvh_hit_linear_miss: u32,
    pub(super) entity_bvh_noprune_mismatches: u32,
    pub(super) entity_bvh_noprune_hit_state_mismatches: u32,
    pub(super) entity_bvh_noprune_distance_mismatches: u32,
    pub(super) entity_bvh_noprune_tetra_mismatches: u32,
    pub(super) entity_bvh_noaabb_mismatches: u32,
    pub(super) entity_bvh_noaabb_hit_state_mismatches: u32,
    pub(super) entity_bvh_noaabb_distance_mismatches: u32,
    pub(super) entity_bvh_noaabb_tetra_mismatches: u32,
    pub(super) entity_linear_order_mismatches: u32,
    pub(super) entity_linear_order_hit_state_mismatches: u32,
    pub(super) entity_linear_order_distance_mismatches: u32,
    pub(super) entity_linear_order_tetra_mismatches: u32,
    pub(super) entity_bvh_leafarray_mismatches: u32,
    pub(super) entity_bvh_leafarray_hit_state_mismatches: u32,
    pub(super) entity_bvh_leafarray_distance_mismatches: u32,
    pub(super) entity_bvh_leafarray_tetra_mismatches: u32,
}

#[derive(Copy, Clone, Default)]
pub(super) struct VteFirstMismatch {
    pub(super) valid: bool,
    pub(super) pixel_x: u32,
    pub(super) pixel_y: u32,
    pub(super) layer: u32,
    pub(super) mismatch_kind: u32,
    pub(super) miss_reason: u32,
    pub(super) debug_flags: u32,
    pub(super) fast_hit: bool,
    pub(super) ref_hit: bool,
    pub(super) fast_chunk: [i32; 4],
    pub(super) ref_chunk: [i32; 4],
    pub(super) fast_material: u32,
    pub(super) ref_material: u32,
    pub(super) fast_hit_t: f32,
    pub(super) ref_hit_t: f32,
    pub(super) chunk_steps_taken: u32,
    pub(super) remaining_voxel_steps: u32,
    pub(super) final_t: f32,
    pub(super) last_chunk: [i32; 4],
}

pub const VTE_MAX_DENSE_CHUNKS: usize = 12_288;
pub(super) const VTE_OCCUPANCY_WORDS_PER_CHUNK: usize = 128; // 8^4 / 32
pub(super) const VTE_MATERIAL_WORDS_PER_CHUNK: usize = 1_024; // 8^4 / 4 packed u8
pub(super) const VTE_MACRO_WORDS_PER_CHUNK: usize = 8; // (8/2)^4 / 32
pub const VTE_REGION_BVH_NODE_CAPACITY: usize = 32_768;
pub const VTE_REGION_LEAF_CAPACITY: usize = 16_384;
pub const VTE_REGION_LEAF_CHUNK_ENTRY_CAPACITY: usize = 262_144;
pub const VTE_REGION_BVH_INVALID_NODE: u32 = u32::MAX;
pub const VTE_REGION_BVH_NODE_FLAG_LEAF: u32 = 1 << 0;
pub const VTE_LEAF_KIND_UNIFORM: u32 = 0;
pub const VTE_LEAF_KIND_VOXEL_CHUNK_ARRAY: u32 = 1;
pub const VTE_LEAF_CHUNK_ENTRY_EMPTY: u32 = 0;
pub const VTE_LEAF_CHUNK_ENTRY_UNIFORM_FLAG: u32 = 1 << 31;
pub(super) const VTE_DEBUG_FLAG_REFERENCE_COMPARE: u32 = 1 << 0;
pub(super) const VTE_DEBUG_FLAG_REFERENCE_MISMATCH_ONLY: u32 = 1 << 1;
pub(super) const VTE_DEBUG_FLAG_COMPARE_SLICE_ONLY: u32 = 1 << 2;
pub(super) const VTE_DEBUG_FLAG_LOD_TINT: u32 = 1 << 3;
pub(super) const VTE_DEBUG_FLAG_ENTITY_LINEAR_ONLY: u32 = 1 << 4;
pub(super) const VTE_DEBUG_FLAG_ENTITY_BVH_COMPARE: u32 = 1 << 5;
pub(super) const VTE_HIGHLIGHT_FLAG_HIT_VOXEL: u32 = 1 << 0;
pub(super) const VTE_HIGHLIGHT_FLAG_PLACE_VOXEL: u32 = 1 << 1;
pub(super) const VTE_COMPARE_STATS_WORD_COUNT: usize = 40;
pub(super) const VTE_COMPARE_STAT_COMPARED: usize = 0;
pub(super) const VTE_COMPARE_STAT_MATCHES: usize = 1;
pub(super) const VTE_COMPARE_STAT_MISMATCHES: usize = 2;
pub(super) const VTE_COMPARE_STAT_HIT_STATE_MISMATCHES: usize = 3;
pub(super) const VTE_COMPARE_STAT_CHUNK_MATERIAL_MISMATCHES: usize = 4;
pub(super) const VTE_COMPARE_STAT_FAST_MISS_REF_HIT: usize = 5;
pub(super) const VTE_COMPARE_STAT_FAST_HIT_REF_MISS: usize = 6;
pub(super) const VTE_COMPARE_STAT_REASON_NONE: usize = 7;
pub(super) const VTE_COMPARE_STAT_REASON_TOUCHED_VISIBLE: usize = 8;
pub(super) const VTE_COMPARE_STAT_REASON_VOXEL_BUDGET: usize = 9;
pub(super) const VTE_COMPARE_STAT_REASON_CHUNK_BUDGET: usize = 10;
pub(super) const VTE_COMPARE_STAT_REASON_MAX_DISTANCE: usize = 11;
pub(super) const VTE_COMPARE_STAT_REASON_LOOKUP_FALSE_NEGATIVE: usize = 12;
pub(super) const VTE_COMPARE_STAT_ZERO_INTERVAL_FLAG: usize = 13;
pub(super) const VTE_COMPARE_STAT_TIE_STEPPED_FLAG: usize = 14;
pub(super) const VTE_COMPARE_STAT_LOOKUP_FALLBACK_FLAG: usize = 15;
pub(super) const VTE_COMPARE_STAT_ENTITY_BVH_SAMPLE: usize = 16;
pub(super) const VTE_COMPARE_STAT_ENTITY_BVH_MISMATCH: usize = 17;
pub(super) const VTE_COMPARE_STAT_ENTITY_BVH_HIT_STATE_MISMATCH: usize = 18;
pub(super) const VTE_COMPARE_STAT_ENTITY_BVH_MATERIAL_MISMATCH: usize = 19;
pub(super) const VTE_COMPARE_STAT_ENTITY_BVH_DISTANCE_MISMATCH: usize = 20;
pub(super) const VTE_COMPARE_STAT_ENTITY_BVH_MISS_LINEAR_HIT: usize = 21;
pub(super) const VTE_COMPARE_STAT_ENTITY_BVH_HIT_LINEAR_MISS: usize = 22;
pub(super) const VTE_COMPARE_STAT_ENTITY_BVH_TETRA_MISMATCH: usize = 23;
pub(super) const VTE_COMPARE_STAT_ENTITY_BVH_NOPRUNE_MISMATCH: usize = 24;
pub(super) const VTE_COMPARE_STAT_ENTITY_BVH_NOPRUNE_HIT_STATE_MISMATCH: usize = 25;
pub(super) const VTE_COMPARE_STAT_ENTITY_BVH_NOPRUNE_DISTANCE_MISMATCH: usize = 26;
pub(super) const VTE_COMPARE_STAT_ENTITY_BVH_NOPRUNE_TETRA_MISMATCH: usize = 27;
pub(super) const VTE_COMPARE_STAT_ENTITY_BVH_NOAABB_MISMATCH: usize = 28;
pub(super) const VTE_COMPARE_STAT_ENTITY_BVH_NOAABB_HIT_STATE_MISMATCH: usize = 29;
pub(super) const VTE_COMPARE_STAT_ENTITY_BVH_NOAABB_DISTANCE_MISMATCH: usize = 30;
pub(super) const VTE_COMPARE_STAT_ENTITY_BVH_NOAABB_TETRA_MISMATCH: usize = 31;
pub(super) const VTE_COMPARE_STAT_ENTITY_LINEAR_ORDER_MISMATCH: usize = 32;
pub(super) const VTE_COMPARE_STAT_ENTITY_LINEAR_ORDER_HIT_STATE_MISMATCH: usize = 33;
pub(super) const VTE_COMPARE_STAT_ENTITY_LINEAR_ORDER_DISTANCE_MISMATCH: usize = 34;
pub(super) const VTE_COMPARE_STAT_ENTITY_LINEAR_ORDER_TETRA_MISMATCH: usize = 35;
pub(super) const VTE_COMPARE_STAT_ENTITY_BVH_LEAFARRAY_MISMATCH: usize = 36;
pub(super) const VTE_COMPARE_STAT_ENTITY_BVH_LEAFARRAY_HIT_STATE_MISMATCH: usize = 37;
pub(super) const VTE_COMPARE_STAT_ENTITY_BVH_LEAFARRAY_DISTANCE_MISMATCH: usize = 38;
pub(super) const VTE_COMPARE_STAT_ENTITY_BVH_LEAFARRAY_TETRA_MISMATCH: usize = 39;
pub(super) const VTE_FIRST_MISMATCH_WORD_COUNT: usize = 27;
pub(super) const VTE_FIRST_MISMATCH_VALID: usize = 0;
pub(super) const VTE_FIRST_MISMATCH_PIXEL_X: usize = 1;
pub(super) const VTE_FIRST_MISMATCH_PIXEL_Y: usize = 2;
pub(super) const VTE_FIRST_MISMATCH_LAYER: usize = 3;
pub(super) const VTE_FIRST_MISMATCH_KIND: usize = 4;
pub(super) const VTE_FIRST_MISMATCH_MISS_REASON: usize = 5;
pub(super) const VTE_FIRST_MISMATCH_DEBUG_FLAGS: usize = 6;
pub(super) const VTE_FIRST_MISMATCH_HIT_MASK: usize = 7;
pub(super) const VTE_FIRST_MISMATCH_FAST_CHUNK_X: usize = 8;
pub(super) const VTE_FIRST_MISMATCH_FAST_CHUNK_Y: usize = 9;
pub(super) const VTE_FIRST_MISMATCH_FAST_CHUNK_Z: usize = 10;
pub(super) const VTE_FIRST_MISMATCH_FAST_CHUNK_W: usize = 11;
pub(super) const VTE_FIRST_MISMATCH_REF_CHUNK_X: usize = 12;
pub(super) const VTE_FIRST_MISMATCH_REF_CHUNK_Y: usize = 13;
pub(super) const VTE_FIRST_MISMATCH_REF_CHUNK_Z: usize = 14;
pub(super) const VTE_FIRST_MISMATCH_REF_CHUNK_W: usize = 15;
pub(super) const VTE_FIRST_MISMATCH_FAST_MATERIAL: usize = 16;
pub(super) const VTE_FIRST_MISMATCH_REF_MATERIAL: usize = 17;
pub(super) const VTE_FIRST_MISMATCH_FAST_HIT_T: usize = 18;
pub(super) const VTE_FIRST_MISMATCH_REF_HIT_T: usize = 19;
pub(super) const VTE_FIRST_MISMATCH_CHUNK_STEPS: usize = 20;
pub(super) const VTE_FIRST_MISMATCH_REMAINING_VOXELS: usize = 21;
pub(super) const VTE_FIRST_MISMATCH_FINAL_T: usize = 22;
pub(super) const VTE_FIRST_MISMATCH_LAST_CHUNK_X: usize = 23;
pub(super) const VTE_FIRST_MISMATCH_LAST_CHUNK_Y: usize = 24;
pub(super) const VTE_FIRST_MISMATCH_LAST_CHUNK_Z: usize = 25;
pub(super) const VTE_FIRST_MISMATCH_LAST_CHUNK_W: usize = 26;
// Always build/use the entity BVH when entity tetrahedra are present.
// This avoids costly per-ray linear tetra loops for small dynamic entity sets.
pub(super) const VTE_ENTITY_LINEAR_THRESHOLD_TETS: usize = 0;

#[inline]
pub(super) fn vte_hash_chunk_coord(chunk_coord: [i32; 4]) -> u32 {
    let x = chunk_coord[0] as u32;
    let y = chunk_coord[1] as u32;
    let z = chunk_coord[2] as u32;
    let w = chunk_coord[3] as u32;
    x.wrapping_mul(0x8DA6_B343)
        ^ y.wrapping_mul(0xD816_3841)
        ^ z.wrapping_mul(0xCB1A_B31F)
        ^ w.wrapping_mul(0x1656_67B1)
}
