use bytemuck::{Pod, Zeroable};
use std::collections::HashSet;
use vulkano::buffer::Subbuffer;

#[derive(Copy, Clone, Debug, Default, Pod, Zeroable)]
#[repr(C)]
pub struct GpuVoxelChunkHeader {
    pub chunk_coord: [i32; 4],
    pub lod_level: u32,
    pub _lod_padding: [u32; 3],
    pub occupancy_word_offset: u32,
    pub material_word_offset: u32,
    pub flags: u32,
    pub macro_word_offset: u32,
    pub solid_local_min: [i32; 4],
    pub solid_local_max: [i32; 4],
}

impl GpuVoxelChunkHeader {
    pub const FLAG_EMPTY: u32 = 1 << 0;
    pub const FLAG_FULL: u32 = 1 << 1;
}

#[derive(Copy, Clone, Debug, Default, Pod, Zeroable)]
#[repr(C)]
pub struct GpuVoxelYSliceBounds {
    pub chunk_y: i32,
    pub lod_level: u32,
    pub min_chunk_x: i32,
    pub max_chunk_x: i32,
    pub min_chunk_z: i32,
    pub max_chunk_z: i32,
    pub min_chunk_w: i32,
    pub max_chunk_w: i32,
    pub lookup_entry_offset: u32,
    pub lookup_entry_count: u32,
    pub lookup_dim_x: u32,
    pub lookup_dim_z: u32,
    pub lookup_dim_w: u32,
    pub _padding: u32,
}

pub struct VoxelFrameInput<'a> {
    pub metadata_generation: u64,
    pub chunk_headers: &'a [GpuVoxelChunkHeader],
    pub payload_update_slots: &'a [u32],
    pub occupancy_words: &'a [u32],
    pub material_words: &'a [u32],
    pub macro_words: &'a [u32],
    pub visible_chunk_indices: &'a [u32],
    pub y_slice_bounds: &'a [GpuVoxelYSliceBounds],
    pub y_slice_lookup_entries: &'a [u32],
}

#[derive(Copy, Clone, Default, Pod, Zeroable)]
#[repr(C)]
pub(super) struct GpuVoxelFrameMeta {
    pub(super) chunk_count: u32,
    pub(super) visible_chunk_count: u32,
    pub(super) occupancy_word_count: u32,
    pub(super) material_word_count: u32,
    pub(super) macro_word_count: u32,
    pub(super) max_trace_steps: u32,
    pub(super) max_trace_distance: f32,
    pub(super) lod_near_max_distance: f32,
    pub(super) lod_mid_max_distance: f32,
    pub(super) chunk_lookup_capacity: u32,
    pub(super) y_slice_count: u32,
    pub(super) y_slice_lookup_entry_count: u32,
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
pub(super) struct GpuVoxelChunkLookupEntry {
    pub(super) chunk_coord: [i32; 4],
    pub(super) chunk_index: u32,
    pub(super) lod_level: u32,
    pub(super) _padding: [u32; 2],
}

impl GpuVoxelChunkLookupEntry {
    pub(super) const INVALID_INDEX: u32 = u32::MAX;

    pub(super) fn empty() -> Self {
        Self {
            chunk_coord: [0; 4],
            chunk_index: Self::INVALID_INDEX,
            lod_level: 0,
            _padding: [0; 2],
        }
    }
}

impl Default for GpuVoxelChunkLookupEntry {
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

pub const VTE_MAX_CHUNKS: usize = 12_288;
pub(super) const VTE_OCCUPANCY_WORDS_PER_CHUNK: usize = 128; // 8^4 / 32
pub(super) const VTE_MATERIAL_WORDS_PER_CHUNK: usize = 1_024; // 8^4 / 4 packed u8
pub(super) const VTE_MACRO_WORDS_PER_CHUNK: usize = 8; // (8/2)^4 / 32
pub(super) const VTE_MAX_Y_SLICES: usize = VTE_MAX_CHUNKS;
pub(super) const VTE_MAX_Y_SLICE_LOOKUP_ENTRIES: usize = 131_072;
pub(super) const VTE_CHUNK_LOOKUP_CAPACITY: usize = 32_768; // Must be power of two.
pub(super) const VTE_DEBUG_FLAG_REFERENCE_COMPARE: u32 = 1 << 0;
pub(super) const VTE_DEBUG_FLAG_REFERENCE_MISMATCH_ONLY: u32 = 1 << 1;
pub(super) const VTE_DEBUG_FLAG_COMPARE_SLICE_ONLY: u32 = 1 << 2;
pub(super) const VTE_DEBUG_FLAG_YSLICE_LOOKUP_CACHE: u32 = 1 << 3;
pub(super) const VTE_DEBUG_FLAG_LOD_TINT: u32 = 1 << 4;
pub(super) const VTE_DEBUG_FLAG_ENTITY_LINEAR_ONLY: u32 = 1 << 5;
pub(super) const VTE_DEBUG_FLAG_ENTITY_BVH_COMPARE: u32 = 1 << 6;
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

#[inline]
pub(super) fn vte_hash_chunk_coord_with_lod(chunk_coord: [i32; 4], lod_level: u32) -> u32 {
    vte_hash_chunk_coord(chunk_coord) ^ lod_level.wrapping_mul(0x9E37_79B9)
}

pub(super) fn stage_voxel_payload_updates(
    input: &VoxelFrameInput<'_>,
    occupancy_cache: &mut [u32],
    material_cache: &mut [u32],
    macro_cache: &mut [u32],
    touched_slots: &mut Vec<u32>,
) -> usize {
    let max_updates = input
        .payload_update_slots
        .len()
        .min(input.occupancy_words.len() / VTE_OCCUPANCY_WORDS_PER_CHUNK)
        .min(input.material_words.len() / VTE_MATERIAL_WORDS_PER_CHUNK)
        .min(input.macro_words.len() / VTE_MACRO_WORDS_PER_CHUNK);

    touched_slots.clear();
    let mut touched_slot_set = HashSet::new();

    for update_idx in 0..max_updates {
        let slot = input.payload_update_slots[update_idx] as usize;
        if slot >= VTE_MAX_CHUNKS {
            continue;
        }

        let src_occ_base = update_idx * VTE_OCCUPANCY_WORDS_PER_CHUNK;
        let src_mat_base = update_idx * VTE_MATERIAL_WORDS_PER_CHUNK;
        let src_macro_base = update_idx * VTE_MACRO_WORDS_PER_CHUNK;
        let dst_occ_base = slot * VTE_OCCUPANCY_WORDS_PER_CHUNK;
        let dst_mat_base = slot * VTE_MATERIAL_WORDS_PER_CHUNK;
        let dst_macro_base = slot * VTE_MACRO_WORDS_PER_CHUNK;

        occupancy_cache[dst_occ_base..dst_occ_base + VTE_OCCUPANCY_WORDS_PER_CHUNK]
            .copy_from_slice(
                &input.occupancy_words[src_occ_base..src_occ_base + VTE_OCCUPANCY_WORDS_PER_CHUNK],
            );
        material_cache[dst_mat_base..dst_mat_base + VTE_MATERIAL_WORDS_PER_CHUNK].copy_from_slice(
            &input.material_words[src_mat_base..src_mat_base + VTE_MATERIAL_WORDS_PER_CHUNK],
        );
        macro_cache[dst_macro_base..dst_macro_base + VTE_MACRO_WORDS_PER_CHUNK].copy_from_slice(
            &input.macro_words[src_macro_base..src_macro_base + VTE_MACRO_WORDS_PER_CHUNK],
        );

        let slot_u32 = slot as u32;
        if touched_slot_set.insert(slot_u32) {
            touched_slots.push(slot_u32);
        }
    }

    max_updates
}

pub(super) fn apply_pending_voxel_payload_updates_for_frame(
    pending_slots: &mut Vec<u32>,
    pending_slot_set: &mut HashSet<u32>,
    voxel_occupancy_words_buffer: &Subbuffer<[u32]>,
    voxel_material_words_buffer: &Subbuffer<[u32]>,
    voxel_macro_words_buffer: &Subbuffer<[u32]>,
    occupancy_cache: &[u32],
    material_cache: &[u32],
    macro_cache: &[u32],
) -> usize {
    let mut slots = std::mem::take(pending_slots);
    pending_slot_set.clear();
    if slots.is_empty() {
        return 0;
    }
    slots.sort_unstable();

    {
        let mut writer = voxel_occupancy_words_buffer.write().unwrap();
        for &slot_u32 in &slots {
            let slot = slot_u32 as usize;
            if slot >= VTE_MAX_CHUNKS {
                continue;
            }
            let base = slot * VTE_OCCUPANCY_WORDS_PER_CHUNK;
            writer[base..base + VTE_OCCUPANCY_WORDS_PER_CHUNK]
                .copy_from_slice(&occupancy_cache[base..base + VTE_OCCUPANCY_WORDS_PER_CHUNK]);
        }
    }
    {
        let mut writer = voxel_material_words_buffer.write().unwrap();
        for &slot_u32 in &slots {
            let slot = slot_u32 as usize;
            if slot >= VTE_MAX_CHUNKS {
                continue;
            }
            let base = slot * VTE_MATERIAL_WORDS_PER_CHUNK;
            writer[base..base + VTE_MATERIAL_WORDS_PER_CHUNK]
                .copy_from_slice(&material_cache[base..base + VTE_MATERIAL_WORDS_PER_CHUNK]);
        }
    }
    {
        let mut writer = voxel_macro_words_buffer.write().unwrap();
        for &slot_u32 in &slots {
            let slot = slot_u32 as usize;
            if slot >= VTE_MAX_CHUNKS {
                continue;
            }
            let base = slot * VTE_MACRO_WORDS_PER_CHUNK;
            writer[base..base + VTE_MACRO_WORDS_PER_CHUNK]
                .copy_from_slice(&macro_cache[base..base + VTE_MACRO_WORDS_PER_CHUNK]);
        }
    }

    slots.len()
}
