use crate::camera::{PLAYER_HEIGHT, PLAYER_RADIUS_XZW};
use crate::voxel::cull::{self, SurfaceData};
use crate::voxel::io as voxel_io;
use crate::voxel::worldgen;
use crate::voxel::{ChunkPos, VoxelType, CHUNK_SIZE, CHUNK_VOLUME};
use higher_dimension_playground::render::{
    GpuVoxelChunkHeader, GpuVoxelYSliceBounds, VoxelFrameInput, VTE_MAX_CHUNKS,
};
use std::collections::{BTreeMap, HashMap, HashSet};
use std::fs::File;
use std::io::{self, BufReader, BufWriter, Write};
use std::path::Path;

const RENDER_DISTANCE: f32 = 64.0;
const OCCUPANCY_WORDS_PER_CHUNK: usize = CHUNK_VOLUME / 32;
const MATERIAL_WORDS_PER_CHUNK: usize = CHUNK_VOLUME / 4; // packed 4x u8 per u32
const MACRO_CELLS_PER_AXIS: usize = CHUNK_SIZE / 2; // 2x2x2x2 macro cells
const MACRO_CELLS_PER_CHUNK: usize =
    MACRO_CELLS_PER_AXIS * MACRO_CELLS_PER_AXIS * MACRO_CELLS_PER_AXIS * MACRO_CELLS_PER_AXIS;
const MACRO_WORDS_PER_CHUNK: usize = MACRO_CELLS_PER_CHUNK / 32;
const Y_SLICE_LOOKUP_MAX_ENTRIES: usize = 131_072;
const Y_SLICE_LOOKUP_MAX_ENTRIES_PER_SLICE: usize = 65_536;
const EDIT_RAY_EPSILON: f32 = 1e-4;
const EDIT_RAY_MAX_STEPS: usize = 4096;
const COLLISION_PUSHUP_STEP: f32 = 0.05;
const COLLISION_MAX_PUSHUP_STEPS: usize = 80;
const COLLISION_BINARY_STEPS: usize = 14;
const HARD_WORLD_FLOOR_Y: f32 = -4.0;
const GPU_PAYLOAD_SLOT_CAPACITY: usize = VTE_MAX_CHUNKS;

struct CachedChunkPayload {
    hash: u64,
    solid_count: u32,
    is_full: bool,
    ref_count: u32,
    gpu_slot: u32,
    occupancy_words: Box<[u32; OCCUPANCY_WORDS_PER_CHUNK]>,
    material_words: Box<[u32; MATERIAL_WORDS_PER_CHUNK]>,
    macro_words: Box<[u32; MACRO_WORDS_PER_CHUNK]>,
    solid_local_min: [i32; 4],
    solid_local_max: [i32; 4],
}

#[derive(Copy, Clone)]
struct ChunkPayloadCacheEntry {
    payload_id: u32,
}

fn chunk_payload_hash(voxels: &[VoxelType; CHUNK_VOLUME], solid_count: u32) -> u64 {
    // FNV-1a over voxel material IDs + solid count.
    let mut hash = 0xcbf2_9ce4_8422_2325u64;
    hash ^= solid_count as u64;
    hash = hash.wrapping_mul(0x0000_0100_0000_01b3);
    for voxel in voxels.iter() {
        hash ^= voxel.0 as u64;
        hash = hash.wrapping_mul(0x0000_0100_0000_01b3);
    }
    hash
}

fn pack_chunk_payload_words(
    chunk: &crate::voxel::chunk::Chunk,
    occupancy_words: &mut [u32],
    material_words: &mut [u32],
    macro_words: &mut [u32],
) -> ([i32; 4], [i32; 4]) {
    occupancy_words.fill(0);
    material_words.fill(0);
    macro_words.fill(0);
    let mut chunk_solid_local_min = [i32::MAX; 4];
    let mut chunk_solid_local_max = [i32::MIN; 4];

    for (voxel_idx, voxel) in chunk.voxels.iter().enumerate() {
        let mat_word_idx = voxel_idx / 4;
        let mat_shift = ((voxel_idx & 3) * 8) as u32;
        material_words[mat_word_idx] |= (voxel.0 as u32) << mat_shift;
        if voxel.is_solid() {
            let word_idx = voxel_idx / 32;
            occupancy_words[word_idx] |= 1u32 << (voxel_idx % 32);

            let x = voxel_idx % CHUNK_SIZE;
            let y = (voxel_idx / CHUNK_SIZE) % CHUNK_SIZE;
            let z = (voxel_idx / (CHUNK_SIZE * CHUNK_SIZE)) % CHUNK_SIZE;
            let w = voxel_idx / (CHUNK_SIZE * CHUNK_SIZE * CHUNK_SIZE);
            chunk_solid_local_min[0] = chunk_solid_local_min[0].min(x as i32);
            chunk_solid_local_min[1] = chunk_solid_local_min[1].min(y as i32);
            chunk_solid_local_min[2] = chunk_solid_local_min[2].min(z as i32);
            chunk_solid_local_min[3] = chunk_solid_local_min[3].min(w as i32);
            chunk_solid_local_max[0] = chunk_solid_local_max[0].max(x as i32);
            chunk_solid_local_max[1] = chunk_solid_local_max[1].max(y as i32);
            chunk_solid_local_max[2] = chunk_solid_local_max[2].max(z as i32);
            chunk_solid_local_max[3] = chunk_solid_local_max[3].max(w as i32);
            let mx = x / 2;
            let my = y / 2;
            let mz = z / 2;
            let mw = w / 2;
            let macro_idx = (((mw * MACRO_CELLS_PER_AXIS + mz) * MACRO_CELLS_PER_AXIS + my)
                * MACRO_CELLS_PER_AXIS)
                + mx;
            let macro_word_idx = macro_idx / 32;
            macro_words[macro_word_idx] |= 1u32 << (macro_idx % 32);
        }
    }

    (
        if chunk_solid_local_min[0] == i32::MAX {
            [0, 0, 0, 0]
        } else {
            chunk_solid_local_min
        },
        if chunk_solid_local_max[0] == i32::MIN {
            [0, 0, 0, 0]
        } else {
            chunk_solid_local_max
        },
    )
}

fn build_cached_chunk_payload(chunk: &crate::voxel::chunk::Chunk) -> CachedChunkPayload {
    let mut occupancy_words = Box::new([0u32; OCCUPANCY_WORDS_PER_CHUNK]);
    let mut material_words = Box::new([0u32; MATERIAL_WORDS_PER_CHUNK]);
    let mut macro_words = Box::new([0u32; MACRO_WORDS_PER_CHUNK]);
    let (solid_local_min, solid_local_max) = pack_chunk_payload_words(
        chunk,
        &mut occupancy_words[..],
        &mut material_words[..],
        &mut macro_words[..],
    );

    CachedChunkPayload {
        hash: chunk_payload_hash(chunk.voxels.as_ref(), chunk.solid_count),
        solid_count: chunk.solid_count,
        is_full: chunk.is_full(),
        ref_count: 0,
        gpu_slot: u32::MAX,
        occupancy_words,
        material_words,
        macro_words,
        solid_local_min,
        solid_local_max,
    }
}

fn cached_chunk_payloads_match(a: &CachedChunkPayload, b: &CachedChunkPayload) -> bool {
    a.solid_count == b.solid_count
        && a.is_full == b.is_full
        && a.solid_local_min == b.solid_local_min
        && a.solid_local_max == b.solid_local_max
        && a.occupancy_words[..] == b.occupancy_words[..]
        && a.material_words[..] == b.material_words[..]
        && a.macro_words[..] == b.macro_words[..]
}

#[derive(Copy, Clone, Debug)]
pub enum ScenePreset {
    Flat,
    DemoCubes,
}

impl ScenePreset {
    fn label(self) -> &'static str {
        match self {
            Self::Flat => "flat",
            Self::DemoCubes => "demo_cubes",
        }
    }
}

pub struct VoxelFrameData {
    pub metadata_generation: u64,
    pub chunk_headers: Vec<GpuVoxelChunkHeader>,
    pub payload_update_slots: Vec<u32>,
    pub occupancy_words: Vec<u32>,
    pub material_words: Vec<u32>,
    pub macro_words: Vec<u32>,
    pub visible_chunk_indices: Vec<u32>,
    pub y_slice_bounds: Vec<GpuVoxelYSliceBounds>,
    pub y_slice_lookup_entries: Vec<u32>,
}

impl VoxelFrameData {
    pub fn as_input(&self) -> VoxelFrameInput<'_> {
        VoxelFrameInput {
            metadata_generation: self.metadata_generation,
            chunk_headers: &self.chunk_headers,
            payload_update_slots: &self.payload_update_slots,
            occupancy_words: &self.occupancy_words,
            material_words: &self.material_words,
            macro_words: &self.macro_words,
            visible_chunk_indices: &self.visible_chunk_indices,
            y_slice_bounds: &self.y_slice_bounds,
            y_slice_lookup_entries: &self.y_slice_lookup_entries,
        }
    }
}

pub struct Scene {
    pub world: crate::voxel::world::VoxelWorld,
    surface: SurfaceData,
    culled_instances: Vec<common::ModelInstance>,
    cull_log_counter: u64,
    voxel_chunk_payload_cache: HashMap<ChunkPos, ChunkPayloadCacheEntry>,
    voxel_chunk_payloads: Vec<Option<CachedChunkPayload>>,
    voxel_chunk_payload_free_ids: Vec<u32>,
    voxel_chunk_payload_hash_buckets: HashMap<u64, Vec<u32>>,
    voxel_payload_slot_to_payload: Vec<Option<u32>>,
    voxel_payload_free_slots: Vec<u32>,
    voxel_pending_payload_uploads: Vec<u32>,
    voxel_pending_payload_upload_set: HashSet<u32>,
    voxel_active_chunks: Vec<ChunkPos>,
    voxel_active_chunk_indices: HashMap<ChunkPos, usize>,
    voxel_world_revision: u64,
    voxel_visibility_generation: u64,
    voxel_cached_visibility_camera_chunk: Option<[i32; 4]>,
    voxel_cached_visibility_world_revision: u64,
    voxel_payload_slot_overflow_logged: bool,
    voxel_frame_data: VoxelFrameData,
}

#[derive(Copy, Clone)]
struct VoxelRayHit {
    solid_voxel: [i32; 4],
    last_empty_voxel: Option<[i32; 4]>,
}

#[derive(Copy, Clone, Debug, Default, PartialEq, Eq)]
pub struct BlockEditTargets {
    pub hit_voxel: Option<[i32; 4]>,
    pub place_voxel: Option<[i32; 4]>,
}

impl Scene {
    fn surface_voxel_count(surface: &SurfaceData) -> u32 {
        surface
            .chunks
            .iter()
            .map(|c| c.voxel_end - c.voxel_start)
            .sum()
    }

    fn rebuild_surface(&mut self, label: &str) {
        self.surface = cull::extract_surfaces(&self.world);
        self.world.clear_dirty();
        let total_voxels = Self::surface_voxel_count(&self.surface);
        eprintln!(
            "{}: {} chunks, {} surface voxels",
            label,
            self.surface.chunks.len(),
            total_voxels
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

    fn queue_payload_upload(&mut self, payload_id: u32) {
        if self.voxel_pending_payload_upload_set.insert(payload_id) {
            self.voxel_pending_payload_uploads.push(payload_id);
        }
    }

    fn add_active_chunk(&mut self, chunk_pos: ChunkPos) {
        if self.voxel_active_chunk_indices.contains_key(&chunk_pos) {
            return;
        }
        let idx = self.voxel_active_chunks.len();
        self.voxel_active_chunks.push(chunk_pos);
        self.voxel_active_chunk_indices.insert(chunk_pos, idx);
    }

    fn remove_active_chunk(&mut self, chunk_pos: ChunkPos) {
        let Some(remove_idx) = self.voxel_active_chunk_indices.remove(&chunk_pos) else {
            return;
        };
        let last_idx = self.voxel_active_chunks.len().saturating_sub(1);
        self.voxel_active_chunks.swap_remove(remove_idx);
        if remove_idx < last_idx {
            let moved = self.voxel_active_chunks[remove_idx];
            self.voxel_active_chunk_indices.insert(moved, remove_idx);
        }
    }

    fn allocate_payload_slot(&mut self) -> Option<u32> {
        if let Some(slot) = self.voxel_payload_free_slots.pop() {
            return Some(slot);
        }
        let next_slot = self.voxel_payload_slot_to_payload.len();
        if next_slot >= GPU_PAYLOAD_SLOT_CAPACITY {
            return None;
        }
        self.voxel_payload_slot_to_payload.push(None);
        Some(next_slot as u32)
    }

    fn intern_cached_chunk_payload(&mut self, payload: CachedChunkPayload) -> Option<u32> {
        let payload_hash = payload.hash;
        let candidate_ids = self
            .voxel_chunk_payload_hash_buckets
            .get(&payload_hash)
            .cloned()
            .unwrap_or_default();

        for payload_id in candidate_ids {
            if let Some(existing_payload) = self
                .voxel_chunk_payloads
                .get_mut(payload_id as usize)
                .and_then(|entry| entry.as_mut())
            {
                if cached_chunk_payloads_match(existing_payload, &payload) {
                    existing_payload.ref_count = existing_payload.ref_count.saturating_add(1);
                    return Some(payload_id);
                }
            }
        }

        let mut payload = payload;
        let Some(gpu_slot) = self.allocate_payload_slot() else {
            if !self.voxel_payload_slot_overflow_logged {
                eprintln!(
                    "VTE payload slot capacity ({GPU_PAYLOAD_SLOT_CAPACITY}) exhausted; skipping new unique payloads."
                );
                self.voxel_payload_slot_overflow_logged = true;
            }
            return None;
        };
        payload.gpu_slot = gpu_slot;
        payload.ref_count = 1;
        let payload_id = if let Some(reused_id) = self.voxel_chunk_payload_free_ids.pop() {
            self.voxel_chunk_payloads[reused_id as usize] = Some(payload);
            reused_id
        } else {
            let new_id = self.voxel_chunk_payloads.len() as u32;
            self.voxel_chunk_payloads.push(Some(payload));
            new_id
        };
        self.voxel_chunk_payload_hash_buckets
            .entry(payload_hash)
            .or_default()
            .push(payload_id);
        self.voxel_payload_slot_to_payload[gpu_slot as usize] = Some(payload_id);
        self.queue_payload_upload(payload_id);
        Some(payload_id)
    }

    fn release_cached_chunk_payload(&mut self, payload_id: u32) {
        let payload_idx = payload_id as usize;
        if payload_idx >= self.voxel_chunk_payloads.len() {
            return;
        }

        let Some(existing_payload) = self.voxel_chunk_payloads[payload_idx].as_mut() else {
            return;
        };
        if existing_payload.ref_count > 1 {
            existing_payload.ref_count -= 1;
            return;
        }
        let payload_hash = existing_payload.hash;
        let gpu_slot = existing_payload.gpu_slot;

        self.voxel_chunk_payloads[payload_idx] = None;
        self.voxel_chunk_payload_free_ids.push(payload_id);
        if self.voxel_pending_payload_upload_set.remove(&payload_id) {
            self.voxel_pending_payload_uploads
                .retain(|&queued_id| queued_id != payload_id);
        }
        if (gpu_slot as usize) < self.voxel_payload_slot_to_payload.len() {
            self.voxel_payload_slot_to_payload[gpu_slot as usize] = None;
            self.voxel_payload_free_slots.push(gpu_slot);
        }

        let mut remove_bucket = false;
        if let Some(bucket) = self.voxel_chunk_payload_hash_buckets.get_mut(&payload_hash) {
            bucket.retain(|&candidate_id| candidate_id != payload_id);
            remove_bucket = bucket.is_empty();
        }
        if remove_bucket {
            self.voxel_chunk_payload_hash_buckets.remove(&payload_hash);
        }
    }

    fn sync_active_chunk_window(&mut self, cam_pos: [f32; 4]) {
        let chunk_radius = (RENDER_DISTANCE / CHUNK_SIZE as f32).ceil() as i32 + 1;
        let cam_chunk = Self::camera_chunk_key(cam_pos);
        let min_chunk = [
            cam_chunk[0] - chunk_radius,
            cam_chunk[1] - chunk_radius,
            cam_chunk[2] - chunk_radius,
            cam_chunk[3] - chunk_radius,
        ];
        let max_chunk = [
            cam_chunk[0] + chunk_radius,
            cam_chunk[1] + chunk_radius,
            cam_chunk[2] + chunk_radius,
            cam_chunk[3] + chunk_radius,
        ];

        let mut required_chunks = Vec::new();
        self.world
            .gather_non_empty_chunks_in_bounds(min_chunk, max_chunk, &mut required_chunks);
        required_chunks.sort_unstable_by_key(|pos| (pos.w, pos.z, pos.y, pos.x));

        let required_set: HashSet<ChunkPos> = required_chunks.iter().copied().collect();
        let mut stale_chunks = Vec::new();
        for &chunk_pos in &self.voxel_active_chunks {
            if !required_set.contains(&chunk_pos) {
                stale_chunks.push(chunk_pos);
            }
        }
        for chunk_pos in stale_chunks {
            if let Some(old_mapping) = self.voxel_chunk_payload_cache.remove(&chunk_pos) {
                self.release_cached_chunk_payload(old_mapping.payload_id);
            }
            self.remove_active_chunk(chunk_pos);
        }

        for chunk_pos in required_chunks {
            if !self.voxel_active_chunk_indices.contains_key(&chunk_pos)
                || !self.voxel_chunk_payload_cache.contains_key(&chunk_pos)
            {
                self.world.queue_chunk_refresh(chunk_pos);
            }
        }
    }

    fn process_queued_voxel_payload_updates(&mut self) {
        let updated_chunks = self.world.drain_pending_chunk_updates();
        if updated_chunks.is_empty() {
            return;
        }

        for chunk_pos in updated_chunks {
            if let Some(old_mapping) = self.voxel_chunk_payload_cache.remove(&chunk_pos) {
                self.release_cached_chunk_payload(old_mapping.payload_id);
            }

            match self.world.chunk_at(chunk_pos) {
                Some(chunk) => {
                    if let Some(payload_id) =
                        self.intern_cached_chunk_payload(build_cached_chunk_payload(chunk))
                    {
                        self.voxel_chunk_payload_cache
                            .insert(chunk_pos, ChunkPayloadCacheEntry { payload_id });
                        self.add_active_chunk(chunk_pos);
                    } else {
                        self.remove_active_chunk(chunk_pos);
                    }
                }
                _ => {
                    self.remove_active_chunk(chunk_pos);
                }
            }
        }

        self.voxel_world_revision = self.voxel_world_revision.wrapping_add(1);
    }

    fn rebuild_visible_voxel_metadata(&mut self, cam_pos: [f32; 4]) {
        struct YSliceBuildData {
            min_chunk_x: i32,
            max_chunk_x: i32,
            min_chunk_z: i32,
            max_chunk_z: i32,
            min_chunk_w: i32,
            max_chunk_w: i32,
            chunk_coords_xzw: Vec<([i32; 3], u32)>,
        }

        let mut chunk_headers = Vec::new();
        let mut visible_chunk_indices = Vec::new();
        let mut y_slice_build: BTreeMap<i32, YSliceBuildData> = BTreeMap::new();
        let mut y_slice_lookup_entries = Vec::new();
        let render_dist_sq = RENDER_DISTANCE * RENDER_DISTANCE;

        for &chunk_pos in &self.voxel_active_chunks {
            // L0 culling: skip chunks outside the same camera radius used by tetra path.
            let chunk_min = [
                chunk_pos.x * CHUNK_SIZE as i32,
                chunk_pos.y * CHUNK_SIZE as i32,
                chunk_pos.z * CHUNK_SIZE as i32,
                chunk_pos.w * CHUNK_SIZE as i32,
            ];
            let chunk_max = [
                chunk_min[0] + CHUNK_SIZE as i32,
                chunk_min[1] + CHUNK_SIZE as i32,
                chunk_min[2] + CHUNK_SIZE as i32,
                chunk_min[3] + CHUNK_SIZE as i32,
            ];

            let mut dist_sq = 0.0f32;
            for d in 0..4 {
                let lo = chunk_min[d] as f32;
                let hi = chunk_max[d] as f32;
                let c = cam_pos[d];
                if c < lo {
                    let delta = lo - c;
                    dist_sq += delta * delta;
                } else if c > hi {
                    let delta = c - hi;
                    dist_sq += delta * delta;
                }
            }
            if dist_sq > render_dist_sq {
                continue;
            }

            let Some(chunk_payload_mapping) = self.voxel_chunk_payload_cache.get(&chunk_pos) else {
                continue;
            };
            let payload_idx = chunk_payload_mapping.payload_id as usize;
            let Some(chunk_payload) = self
                .voxel_chunk_payloads
                .get(payload_idx)
                .and_then(|entry| entry.as_ref())
            else {
                continue;
            };

            let slot = chunk_payload.gpu_slot as usize;
            if slot >= GPU_PAYLOAD_SLOT_CAPACITY {
                continue;
            }
            let occupancy_word_offset = (slot * OCCUPANCY_WORDS_PER_CHUNK) as u32;
            let material_word_offset = (slot * MATERIAL_WORDS_PER_CHUNK) as u32;
            let macro_word_offset = (slot * MACRO_WORDS_PER_CHUNK) as u32;

            let mut flags = 0u32;
            if chunk_payload.is_full {
                flags |= GpuVoxelChunkHeader::FLAG_FULL;
            }

            let chunk_index = chunk_headers.len() as u32;
            chunk_headers.push(GpuVoxelChunkHeader {
                chunk_coord: [chunk_pos.x, chunk_pos.y, chunk_pos.z, chunk_pos.w],
                occupancy_word_offset,
                material_word_offset,
                flags,
                macro_word_offset,
                solid_local_min: chunk_payload.solid_local_min,
                solid_local_max: chunk_payload.solid_local_max,
            });
            visible_chunk_indices.push(chunk_index);

            y_slice_build
                .entry(chunk_pos.y)
                .and_modify(|slice| {
                    slice.min_chunk_x = slice.min_chunk_x.min(chunk_pos.x);
                    slice.max_chunk_x = slice.max_chunk_x.max(chunk_pos.x);
                    slice.min_chunk_z = slice.min_chunk_z.min(chunk_pos.z);
                    slice.max_chunk_z = slice.max_chunk_z.max(chunk_pos.z);
                    slice.min_chunk_w = slice.min_chunk_w.min(chunk_pos.w);
                    slice.max_chunk_w = slice.max_chunk_w.max(chunk_pos.w);
                    slice
                        .chunk_coords_xzw
                        .push(([chunk_pos.x, chunk_pos.z, chunk_pos.w], chunk_index));
                })
                .or_insert_with(|| YSliceBuildData {
                    min_chunk_x: chunk_pos.x,
                    max_chunk_x: chunk_pos.x,
                    min_chunk_z: chunk_pos.z,
                    max_chunk_z: chunk_pos.z,
                    min_chunk_w: chunk_pos.w,
                    max_chunk_w: chunk_pos.w,
                    chunk_coords_xzw: vec![([chunk_pos.x, chunk_pos.z, chunk_pos.w], chunk_index)],
                });
        }

        let mut y_slice_bounds = Vec::with_capacity(y_slice_build.len());
        for (chunk_y, slice) in y_slice_build {
            let dim_x = (slice.max_chunk_x - slice.min_chunk_x + 1).max(0) as usize;
            let dim_z = (slice.max_chunk_z - slice.min_chunk_z + 1).max(0) as usize;
            let dim_w = (slice.max_chunk_w - slice.min_chunk_w + 1).max(0) as usize;
            let volume = dim_x
                .checked_mul(dim_z)
                .and_then(|v| v.checked_mul(dim_w))
                .unwrap_or(0);
            let mut lookup_entry_offset = 0u32;
            let mut lookup_entry_count = 0u32;
            if volume > 0 && volume <= Y_SLICE_LOOKUP_MAX_ENTRIES_PER_SLICE {
                if y_slice_lookup_entries.len().saturating_add(volume) <= Y_SLICE_LOOKUP_MAX_ENTRIES
                {
                    lookup_entry_offset = y_slice_lookup_entries.len() as u32;
                    lookup_entry_count = volume as u32;
                    y_slice_lookup_entries.resize(y_slice_lookup_entries.len() + volume, 0);
                    for ([chunk_x, chunk_z, chunk_w], chunk_index) in slice.chunk_coords_xzw {
                        let local_x = (chunk_x - slice.min_chunk_x) as usize;
                        let local_z = (chunk_z - slice.min_chunk_z) as usize;
                        let local_w = (chunk_w - slice.min_chunk_w) as usize;
                        let linear =
                            ((local_w * dim_z + local_z) * dim_x + local_x).min(volume - 1);
                        let entry_idx = (lookup_entry_offset as usize) + linear;
                        // 0 means "no chunk", so store chunk indices + 1.
                        y_slice_lookup_entries[entry_idx] = chunk_index.saturating_add(1);
                    }
                }
            }

            y_slice_bounds.push(GpuVoxelYSliceBounds {
                chunk_y,
                min_chunk_x: slice.min_chunk_x,
                max_chunk_x: slice.max_chunk_x,
                min_chunk_z: slice.min_chunk_z,
                max_chunk_z: slice.max_chunk_z,
                min_chunk_w: slice.min_chunk_w,
                max_chunk_w: slice.max_chunk_w,
                lookup_entry_offset,
                lookup_entry_count,
                lookup_dim_x: dim_x as u32,
                lookup_dim_z: dim_z as u32,
                lookup_dim_w: dim_w as u32,
                _padding: 0,
            });
        }

        self.voxel_visibility_generation = self.voxel_visibility_generation.wrapping_add(1);
        self.voxel_frame_data.metadata_generation = self.voxel_visibility_generation;
        self.voxel_frame_data.chunk_headers = chunk_headers;
        self.voxel_frame_data.visible_chunk_indices = visible_chunk_indices;
        self.voxel_frame_data.y_slice_bounds = y_slice_bounds;
        self.voxel_frame_data.y_slice_lookup_entries = y_slice_lookup_entries;
    }

    fn rebuild_pending_payload_upload_words(&mut self) {
        let pending_ids = std::mem::take(&mut self.voxel_pending_payload_uploads);
        self.voxel_pending_payload_upload_set.clear();

        let mut payload_update_slots = Vec::new();
        let mut occupancy_words = Vec::new();
        let mut material_words = Vec::new();
        let mut macro_words = Vec::new();
        payload_update_slots.reserve(pending_ids.len());
        occupancy_words.reserve(pending_ids.len() * OCCUPANCY_WORDS_PER_CHUNK);
        material_words.reserve(pending_ids.len() * MATERIAL_WORDS_PER_CHUNK);
        macro_words.reserve(pending_ids.len() * MACRO_WORDS_PER_CHUNK);

        for payload_id in pending_ids {
            let Some(payload) = self
                .voxel_chunk_payloads
                .get(payload_id as usize)
                .and_then(|entry| entry.as_ref())
            else {
                continue;
            };
            if (payload.gpu_slot as usize) >= GPU_PAYLOAD_SLOT_CAPACITY {
                continue;
            }
            payload_update_slots.push(payload.gpu_slot);
            occupancy_words.extend_from_slice(&payload.occupancy_words[..]);
            material_words.extend_from_slice(&payload.material_words[..]);
            macro_words.extend_from_slice(&payload.macro_words[..]);
        }

        self.voxel_frame_data.payload_update_slots = payload_update_slots;
        self.voxel_frame_data.occupancy_words = occupancy_words;
        self.voxel_frame_data.material_words = material_words;
        self.voxel_frame_data.macro_words = macro_words;
    }

    pub fn new(preset: ScenePreset) -> Self {
        let world = match preset {
            ScenePreset::Flat => worldgen::generate_flat_world(
                5,             // 5×5×5 chunks in X, Z, W
                VoxelType(11), // neutral grid floor
            ),
            ScenePreset::DemoCubes => worldgen::generate_demo_cube_layout_world(),
        };

        let surface = cull::extract_surfaces(&world);
        let scene = Self {
            world,
            surface,
            culled_instances: Vec::new(),
            cull_log_counter: 0,
            voxel_chunk_payload_cache: HashMap::new(),
            voxel_chunk_payloads: Vec::new(),
            voxel_chunk_payload_free_ids: Vec::new(),
            voxel_chunk_payload_hash_buckets: HashMap::new(),
            voxel_payload_slot_to_payload: Vec::new(),
            voxel_payload_free_slots: Vec::new(),
            voxel_pending_payload_uploads: Vec::new(),
            voxel_pending_payload_upload_set: HashSet::new(),
            voxel_active_chunks: Vec::new(),
            voxel_active_chunk_indices: HashMap::new(),
            voxel_world_revision: 0,
            voxel_visibility_generation: 0,
            voxel_cached_visibility_camera_chunk: None,
            voxel_cached_visibility_world_revision: 0,
            voxel_payload_slot_overflow_logged: false,
            voxel_frame_data: VoxelFrameData {
                metadata_generation: 0,
                chunk_headers: Vec::new(),
                payload_update_slots: Vec::new(),
                occupancy_words: Vec::new(),
                material_words: Vec::new(),
                macro_words: Vec::new(),
                visible_chunk_indices: Vec::new(),
                y_slice_bounds: Vec::new(),
                y_slice_lookup_entries: Vec::new(),
            },
        };
        let total_voxels = Self::surface_voxel_count(&scene.surface);
        eprintln!(
            "Voxel surface ({}): {} chunks, {} surface voxels",
            preset.label(),
            scene.surface.chunks.len(),
            total_voxels
        );
        scene
    }

    pub fn replace_world(&mut self, world: crate::voxel::world::VoxelWorld) {
        self.world = world;
        self.voxel_chunk_payload_cache.clear();
        self.voxel_chunk_payloads.clear();
        self.voxel_chunk_payload_free_ids.clear();
        self.voxel_chunk_payload_hash_buckets.clear();
        self.voxel_payload_slot_to_payload.clear();
        self.voxel_payload_free_slots.clear();
        self.voxel_pending_payload_uploads.clear();
        self.voxel_pending_payload_upload_set.clear();
        self.voxel_active_chunks.clear();
        self.voxel_active_chunk_indices.clear();
        self.voxel_world_revision = 0;
        self.voxel_visibility_generation = 0;
        self.voxel_cached_visibility_camera_chunk = None;
        self.voxel_cached_visibility_world_revision = 0;
        self.voxel_payload_slot_overflow_logged = false;
        self.voxel_frame_data.metadata_generation = 0;
        self.voxel_frame_data.chunk_headers.clear();
        self.voxel_frame_data.payload_update_slots.clear();
        self.voxel_frame_data.occupancy_words.clear();
        self.voxel_frame_data.material_words.clear();
        self.voxel_frame_data.macro_words.clear();
        self.voxel_frame_data.visible_chunk_indices.clear();
        self.voxel_frame_data.y_slice_bounds.clear();
        self.voxel_frame_data.y_slice_lookup_entries.clear();
        self.rebuild_surface("Voxel surface (loaded)");
    }

    pub fn save_world_to_path(&self, path: &Path) -> io::Result<usize> {
        if let Some(parent) = path.parent() {
            if !parent.as_os_str().is_empty() {
                std::fs::create_dir_all(parent)?;
            }
        }

        let mut writer = BufWriter::new(File::create(path)?);
        voxel_io::save_world(&self.world, &mut writer)?;
        writer.flush()?;

        Ok(self
            .world
            .chunks
            .values()
            .filter(|chunk| !chunk.is_empty())
            .count())
    }

    pub fn load_world_from_path(&mut self, path: &Path) -> io::Result<usize> {
        let mut reader = BufReader::new(File::open(path)?);
        let world = voxel_io::load_world(&mut reader)?;
        let non_empty_chunks = world
            .chunks
            .values()
            .filter(|chunk| !chunk.is_empty())
            .count();
        self.replace_world(world);
        Ok(non_empty_chunks)
    }

    /// Rebuild surface data if any chunk is dirty.
    pub fn update_surfaces_if_dirty(&mut self) {
        if self.world.any_dirty() {
            self.rebuild_surface("Voxel surface rebuilt");
        }
    }

    /// Per-frame: cull and build ModelInstances for the current camera position.
    pub fn build_instances(&mut self, cam_pos: [f32; 4]) -> &[common::ModelInstance] {
        self.culled_instances.clear();
        cull::cull_and_build(
            &self.surface,
            cam_pos,
            RENDER_DISTANCE,
            &mut self.culled_instances,
        );

        let (num_instances, num_tets) = cull::mesh_stats(&self.culled_instances);
        if self.cull_log_counter == 0 || self.cull_log_counter % 120 == 0 {
            eprintln!("Culled: {num_instances} instances, {num_tets} tetrahedra");
        }
        self.cull_log_counter = self.cull_log_counter.wrapping_add(1);

        &self.culled_instances
    }

    /// Build the voxel-native frame payload for VTE.
    pub fn build_voxel_frame_data(&mut self, cam_pos: [f32; 4]) -> &VoxelFrameData {
        self.sync_active_chunk_window(cam_pos);
        self.process_queued_voxel_payload_updates();

        let cam_chunk = Self::camera_chunk_key(cam_pos);
        let visibility_cache_valid = self.voxel_cached_visibility_camera_chunk == Some(cam_chunk)
            && self.voxel_cached_visibility_world_revision == self.voxel_world_revision;
        if !visibility_cache_valid {
            self.rebuild_visible_voxel_metadata(cam_pos);
            self.voxel_cached_visibility_camera_chunk = Some(cam_chunk);
            self.voxel_cached_visibility_world_revision = self.voxel_world_revision;
        }

        self.rebuild_pending_payload_upload_words();
        &self.voxel_frame_data
    }

    fn player_aabb(pos: [f32; 4]) -> ([f32; 4], [f32; 4]) {
        let min = [
            pos[0] - PLAYER_RADIUS_XZW,
            pos[1] - PLAYER_HEIGHT,
            pos[2] - PLAYER_RADIUS_XZW,
            pos[3] - PLAYER_RADIUS_XZW,
        ];
        let max = [
            pos[0] + PLAYER_RADIUS_XZW,
            pos[1],
            pos[2] + PLAYER_RADIUS_XZW,
            pos[3] + PLAYER_RADIUS_XZW,
        ];
        (min, max)
    }

    fn aabb_intersects_solid(&self, min: [f32; 4], max: [f32; 4]) -> bool {
        // Infinite hard floor: disallow moving below this world Y plane.
        if min[1] < HARD_WORLD_FLOOR_Y {
            return true;
        }

        let max_epsilon = 1e-4f32;
        let lo = [
            min[0].floor() as i32,
            min[1].floor() as i32,
            min[2].floor() as i32,
            min[3].floor() as i32,
        ];
        let hi = [
            (max[0] - max_epsilon).floor() as i32,
            (max[1] - max_epsilon).floor() as i32,
            (max[2] - max_epsilon).floor() as i32,
            (max[3] - max_epsilon).floor() as i32,
        ];
        if hi[0] < lo[0] || hi[1] < lo[1] || hi[2] < lo[2] || hi[3] < lo[3] {
            return false;
        }

        for x in lo[0]..=hi[0] {
            for y in lo[1]..=hi[1] {
                for z in lo[2]..=hi[2] {
                    for w in lo[3]..=hi[3] {
                        if self.world.get_voxel(x, y, z, w).is_solid() {
                            return true;
                        }
                    }
                }
            }
        }
        false
    }

    fn player_collides_at(&self, pos: [f32; 4]) -> bool {
        let (min, max) = Self::player_aabb(pos);
        self.aabb_intersects_solid(min, max)
    }

    fn depenetrate_upward(&self, pos: &mut [f32; 4]) {
        if !self.player_collides_at(*pos) {
            return;
        }
        for _ in 0..COLLISION_MAX_PUSHUP_STEPS {
            pos[1] += COLLISION_PUSHUP_STEP;
            if !self.player_collides_at(*pos) {
                return;
            }
        }
    }

    /// Resolve player movement against solid voxels.
    /// Returns `(resolved_position, grounded)`.
    pub fn resolve_player_collision(
        &self,
        old_pos: [f32; 4],
        attempted_pos: [f32; 4],
        velocity_y: &mut f32,
    ) -> ([f32; 4], bool) {
        let mut pos = old_pos;
        self.depenetrate_upward(&mut pos);

        for axis in [0usize, 2, 3, 1] {
            let target = attempted_pos[axis];
            if (target - pos[axis]).abs() <= 1e-6 {
                continue;
            }

            let mut candidate = pos;
            candidate[axis] = target;
            if !self.player_collides_at(candidate) {
                pos = candidate;
                continue;
            }

            let mut feasible = pos[axis];
            let mut blocked = target;
            for _ in 0..COLLISION_BINARY_STEPS {
                let mid = 0.5 * (feasible + blocked);
                let mut probe = pos;
                probe[axis] = mid;
                if self.player_collides_at(probe) {
                    blocked = mid;
                } else {
                    feasible = mid;
                }
            }
            pos[axis] = feasible;

            if axis == 1 {
                *velocity_y = 0.0;
            }
        }

        let mut ground_probe = pos;
        ground_probe[1] -= 0.02;
        let grounded = self.player_collides_at(ground_probe);
        if grounded && *velocity_y < 0.0 {
            *velocity_y = 0.0;
        }

        (pos, grounded)
    }

    fn trace_first_solid_voxel(
        &self,
        ray_origin: [f32; 4],
        ray_direction: [f32; 4],
        max_distance: f32,
    ) -> Option<VoxelRayHit> {
        let dir_len_sq = ray_direction[0] * ray_direction[0]
            + ray_direction[1] * ray_direction[1]
            + ray_direction[2] * ray_direction[2]
            + ray_direction[3] * ray_direction[3];
        if dir_len_sq <= 1e-8 || max_distance <= 0.0 {
            return None;
        }

        let inv_dir_len = dir_len_sq.sqrt().recip();
        let dir = [
            ray_direction[0] * inv_dir_len,
            ray_direction[1] * inv_dir_len,
            ray_direction[2] * inv_dir_len,
            ray_direction[3] * inv_dir_len,
        ];
        let origin = [
            ray_origin[0] + dir[0] * EDIT_RAY_EPSILON,
            ray_origin[1] + dir[1] * EDIT_RAY_EPSILON,
            ray_origin[2] + dir[2] * EDIT_RAY_EPSILON,
            ray_origin[3] + dir[3] * EDIT_RAY_EPSILON,
        ];

        let mut cell = [
            origin[0].floor() as i32,
            origin[1].floor() as i32,
            origin[2].floor() as i32,
            origin[3].floor() as i32,
        ];

        let mut step = [0i32; 4];
        let mut t_max = [f32::INFINITY; 4];
        let mut t_delta = [f32::INFINITY; 4];

        for axis in 0..4 {
            let d = dir[axis];
            if d > 1e-8 {
                step[axis] = 1;
                let next_boundary = cell[axis] as f32 + 1.0;
                t_max[axis] = (next_boundary - origin[axis]) / d;
                t_delta[axis] = 1.0 / d;
            } else if d < -1e-8 {
                step[axis] = -1;
                let next_boundary = cell[axis] as f32;
                t_max[axis] = (next_boundary - origin[axis]) / d;
                t_delta[axis] = -1.0 / d;
            }
        }

        let mut last_empty_voxel = None;
        let v0 = self.world.get_voxel(cell[0], cell[1], cell[2], cell[3]);
        if v0.is_solid() {
            return Some(VoxelRayHit {
                solid_voxel: cell,
                last_empty_voxel,
            });
        }
        last_empty_voxel = Some(cell);

        for _ in 0..EDIT_RAY_MAX_STEPS {
            let mut axis = 0usize;
            for candidate in 1..4 {
                if t_max[candidate] < t_max[axis] {
                    axis = candidate;
                }
            }

            let traversed_t = t_max[axis];
            if !traversed_t.is_finite() || traversed_t > max_distance {
                break;
            }

            cell[axis] += step[axis];
            t_max[axis] += t_delta[axis];

            let voxel = self.world.get_voxel(cell[0], cell[1], cell[2], cell[3]);
            if voxel.is_solid() {
                return Some(VoxelRayHit {
                    solid_voxel: cell,
                    last_empty_voxel,
                });
            }
            last_empty_voxel = Some(cell);
        }

        None
    }

    /// Remove the first solid voxel intersected by a camera ray.
    pub fn remove_block_along_ray(
        &mut self,
        ray_origin: [f32; 4],
        ray_direction: [f32; 4],
        max_distance: f32,
    ) -> Option<[i32; 4]> {
        let hit = self.trace_first_solid_voxel(ray_origin, ray_direction, max_distance)?;
        let [x, y, z, w] = hit.solid_voxel;
        self.world.set_voxel(x, y, z, w, VoxelType::AIR);
        Some(hit.solid_voxel)
    }

    /// Query edit ray targets without mutating the world.
    pub fn block_edit_targets(
        &self,
        ray_origin: [f32; 4],
        ray_direction: [f32; 4],
        max_distance: f32,
    ) -> BlockEditTargets {
        match self.trace_first_solid_voxel(ray_origin, ray_direction, max_distance) {
            Some(hit) => BlockEditTargets {
                hit_voxel: Some(hit.solid_voxel),
                place_voxel: hit.last_empty_voxel,
            },
            None => BlockEditTargets::default(),
        }
    }

    /// Place a voxel in the last empty cell before the first solid hit.
    pub fn place_block_along_ray(
        &mut self,
        ray_origin: [f32; 4],
        ray_direction: [f32; 4],
        max_distance: f32,
        material: VoxelType,
    ) -> Option<[i32; 4]> {
        if material.is_air() {
            return None;
        }
        let hit = self.trace_first_solid_voxel(ray_origin, ray_direction, max_distance)?;
        let target = hit.last_empty_voxel?;
        let [x, y, z, w] = target;
        if self.world.get_voxel(x, y, z, w).is_solid() {
            return None;
        }
        self.world.set_voxel(x, y, z, w, material);
        Some(target)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::voxel::world::VoxelWorld;

    fn make_scene_with_world(world: VoxelWorld) -> Scene {
        let surface = cull::extract_surfaces(&world);
        Scene {
            world,
            surface,
            culled_instances: Vec::new(),
            cull_log_counter: 0,
            voxel_chunk_payload_cache: std::collections::HashMap::new(),
            voxel_chunk_payloads: Vec::new(),
            voxel_chunk_payload_free_ids: Vec::new(),
            voxel_chunk_payload_hash_buckets: std::collections::HashMap::new(),
            voxel_payload_slot_to_payload: Vec::new(),
            voxel_payload_free_slots: Vec::new(),
            voxel_pending_payload_uploads: Vec::new(),
            voxel_pending_payload_upload_set: std::collections::HashSet::new(),
            voxel_active_chunks: Vec::new(),
            voxel_active_chunk_indices: std::collections::HashMap::new(),
            voxel_world_revision: 0,
            voxel_visibility_generation: 0,
            voxel_cached_visibility_camera_chunk: None,
            voxel_cached_visibility_world_revision: 0,
            voxel_payload_slot_overflow_logged: false,
            voxel_frame_data: VoxelFrameData {
                metadata_generation: 0,
                chunk_headers: Vec::new(),
                payload_update_slots: Vec::new(),
                occupancy_words: Vec::new(),
                material_words: Vec::new(),
                macro_words: Vec::new(),
                visible_chunk_indices: Vec::new(),
                y_slice_bounds: Vec::new(),
                y_slice_lookup_entries: Vec::new(),
            },
        }
    }

    #[test]
    fn remove_block_along_ray_hits_first_solid_voxel() {
        let mut world = VoxelWorld::new();
        world.set_voxel(0, 0, 0, 0, VoxelType(7));
        let mut scene = make_scene_with_world(world);

        let removed =
            scene.remove_block_along_ray([0.5, 0.5, -2.0, 0.5], [0.0, 0.0, 1.0, 0.0], 8.0);

        assert_eq!(removed, Some([0, 0, 0, 0]));
        assert!(scene.world.get_voxel(0, 0, 0, 0).is_air());
    }

    #[test]
    fn place_block_along_ray_places_in_last_empty_before_hit() {
        let mut world = VoxelWorld::new();
        world.set_voxel(0, 0, 0, 0, VoxelType(7));
        let mut scene = make_scene_with_world(world);

        let placed = scene.place_block_along_ray(
            [0.5, 0.5, -2.0, 0.5],
            [0.0, 0.0, 1.0, 0.0],
            8.0,
            VoxelType(3),
        );

        assert_eq!(placed, Some([0, 0, -1, 0]));
        assert!(scene.world.get_voxel(0, 0, -1, 0) == VoxelType(3));
        assert!(scene.world.get_voxel(0, 0, 0, 0) == VoxelType(7));
    }

    #[test]
    fn block_edit_targets_reports_hit_and_placement_voxels() {
        let mut world = VoxelWorld::new();
        world.set_voxel(0, 0, 0, 0, VoxelType(7));
        let scene = make_scene_with_world(world);

        let targets = scene.block_edit_targets([0.5, 0.5, -2.0, 0.5], [0.0, 0.0, 1.0, 0.0], 8.0);

        assert_eq!(
            targets,
            BlockEditTargets {
                hit_voxel: Some([0, 0, 0, 0]),
                place_voxel: Some([0, 0, -1, 0]),
            }
        );
    }

    #[test]
    fn resolve_player_collision_lands_on_voxel_surface() {
        let mut world = VoxelWorld::new();
        world.set_voxel(0, -1, 0, 0, VoxelType(3));
        let scene = make_scene_with_world(world);

        let old_pos = [0.5, 2.4, 0.5, 0.5];
        let attempted = [0.5, 1.1, 0.5, 0.5];
        let mut velocity_y = -5.0;
        let (resolved, grounded) =
            scene.resolve_player_collision(old_pos, attempted, &mut velocity_y);

        assert!(grounded);
        assert!((resolved[1] - PLAYER_HEIGHT).abs() < 0.02);
        assert_eq!(velocity_y, 0.0);
    }

    #[test]
    fn resolve_player_collision_blocks_horizontal_motion() {
        let mut world = VoxelWorld::new();
        world.set_voxel(1, 0, 0, 0, VoxelType(3));
        let scene = make_scene_with_world(world);

        let old_pos = [0.35, 1.2, 0.5, 0.5];
        let attempted = [1.6, 1.2, 0.5, 0.5];
        let mut velocity_y = 0.0;
        let (resolved, _grounded) =
            scene.resolve_player_collision(old_pos, attempted, &mut velocity_y);

        // Player radius keeps center below x=0.7 when blocked by voxel slab at x>=1.
        assert!(resolved[0] < 0.72);
    }

    #[test]
    fn resolve_player_collision_lands_on_hard_world_floor() {
        let world = VoxelWorld::new();
        let scene = make_scene_with_world(world);

        let old_pos = [0.0, 0.4, 0.0, 0.0];
        let attempted = [0.0, -6.0, 0.0, 0.0];
        let mut velocity_y = -9.0;
        let (resolved, grounded) =
            scene.resolve_player_collision(old_pos, attempted, &mut velocity_y);

        assert!(grounded);
        assert!((resolved[1] - (HARD_WORLD_FLOOR_Y + PLAYER_HEIGHT)).abs() < 0.03);
        assert_eq!(velocity_y, 0.0);
    }

    #[test]
    fn replace_world_rebuilds_surface_for_clean_loaded_world() {
        let mut world = VoxelWorld::new();
        world.set_voxel(0, 0, 0, 0, VoxelType(3));
        let mut scene = make_scene_with_world(world);
        assert!(!scene.surface.chunks.is_empty());

        scene.replace_world(VoxelWorld::new());
        assert!(scene.surface.chunks.is_empty());
    }
}
