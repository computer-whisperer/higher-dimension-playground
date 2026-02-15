use super::*;

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

    fn camera_chunk_key_for_scale(cam_pos: [f32; 4], cell_scale: i32) -> [i32; 4] {
        let span = (CHUNK_SIZE as i32).saturating_mul(cell_scale.max(1));
        [
            (cam_pos[0].floor() as i32).div_euclid(span),
            (cam_pos[1].floor() as i32).div_euclid(span),
            (cam_pos[2].floor() as i32).div_euclid(span),
            (cam_pos[3].floor() as i32).div_euclid(span),
        ]
    }

    fn lod_chunk_world_scale(lod_level: u8) -> f32 {
        match lod_level {
            VOXEL_LOD_LEVEL_NEAR => 1.0,
            VOXEL_LOD_LEVEL_MID => 2.0,
            VOXEL_LOD_LEVEL_FAR => 4.0,
            _ => 1.0,
        }
    }

    fn chunk_distance_bounds_sq(cam_pos: [f32; 4], key: RuntimeChunkKey) -> (f32, f32) {
        let lod_scale = Self::lod_chunk_world_scale(key.lod_level);
        let span = (CHUNK_SIZE as f32) * lod_scale;
        let min = [
            (key.chunk_pos.x as f32) * span,
            (key.chunk_pos.y as f32) * span,
            (key.chunk_pos.z as f32) * span,
            (key.chunk_pos.w as f32) * span,
        ];
        let max = [min[0] + span, min[1] + span, min[2] + span, min[3] + span];

        let mut min_dist_sq = 0.0f32;
        let mut max_dist_sq = 0.0f32;
        for axis in 0..4 {
            let p = cam_pos[axis];
            let d_min = if p < min[axis] {
                min[axis] - p
            } else if p > max[axis] {
                p - max[axis]
            } else {
                0.0
            };
            min_dist_sq += d_min * d_min;

            let d0 = (p - min[axis]).abs();
            let d1 = (p - max[axis]).abs();
            let d_far = d0.max(d1);
            max_dist_sq += d_far * d_far;
        }
        (min_dist_sq, max_dist_sq)
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

    fn queue_payload_upload(&mut self, payload_id: u32) {
        if self.voxel_pending_payload_upload_set.insert(payload_id) {
            self.voxel_pending_payload_uploads.push(payload_id);
        }
    }

    fn add_active_chunk(&mut self, key: RuntimeChunkKey) {
        if self.voxel_active_chunk_indices.contains_key(&key) {
            return;
        }
        let idx = self.voxel_active_chunks.len();
        self.voxel_active_chunks.push(key);
        self.voxel_active_chunk_indices.insert(key, idx);
    }

    fn remove_active_chunk(&mut self, key: RuntimeChunkKey) {
        let Some(remove_idx) = self.voxel_active_chunk_indices.remove(&key) else {
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

    fn sync_active_chunk_window(&mut self, cam_pos: [f32; 4], near_lod_max_distance: f32) {
        let near_active_distance = near_lod_max_distance.max(VOXEL_NEAR_ACTIVE_DISTANCE);
        let chunk_radius = (near_active_distance / CHUNK_SIZE as f32).ceil() as i32 + 1;
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
        let mut required_keys: Vec<RuntimeChunkKey> = required_chunks
            .into_iter()
            .map(|chunk_pos| RuntimeChunkKey {
                lod_level: VOXEL_LOD_LEVEL_NEAR,
                chunk_pos,
            })
            .collect();
        required_keys.extend(self.voxel_lod_chunks.keys().copied());
        required_keys.sort_unstable_by_key(|key| {
            (
                key.lod_level,
                key.chunk_pos.w,
                key.chunk_pos.z,
                key.chunk_pos.y,
                key.chunk_pos.x,
            )
        });

        let required_set: HashSet<RuntimeChunkKey> = required_keys.iter().copied().collect();
        let mut stale_chunks = Vec::new();
        for &key in &self.voxel_active_chunks {
            if !required_set.contains(&key) {
                stale_chunks.push(key);
            }
        }
        for key in stale_chunks {
            if let Some(old_mapping) = self.voxel_chunk_payload_cache.remove(&key) {
                self.release_cached_chunk_payload(old_mapping.payload_id);
            }
            self.remove_active_chunk(key);
        }

        for key in required_keys {
            if !self.voxel_active_chunk_indices.contains_key(&key)
                || !self.voxel_chunk_payload_cache.contains_key(&key)
            {
                if key.lod_level == VOXEL_LOD_LEVEL_NEAR {
                    self.world.queue_chunk_refresh(key.chunk_pos);
                } else {
                    self.queue_lod_chunk_update(key);
                }
            }
        }
    }

    fn process_queued_voxel_payload_updates(&mut self) {
        let mut updated_chunks: Vec<RuntimeChunkKey> = self
            .world
            .drain_pending_chunk_updates()
            .into_iter()
            .map(|chunk_pos| RuntimeChunkKey {
                lod_level: VOXEL_LOD_LEVEL_NEAR,
                chunk_pos,
            })
            .collect();
        let mut lod_updates = std::mem::take(&mut self.voxel_pending_lod_chunk_updates);
        self.voxel_pending_lod_chunk_update_set.clear();
        updated_chunks.append(&mut lod_updates);
        if updated_chunks.is_empty() {
            return;
        }

        for key in updated_chunks {
            if let Some(old_mapping) = self.voxel_chunk_payload_cache.remove(&key) {
                self.release_cached_chunk_payload(old_mapping.payload_id);
            }

            let chunk = if key.lod_level == VOXEL_LOD_LEVEL_NEAR {
                self.world.chunk_at(key.chunk_pos)
            } else {
                self.voxel_lod_chunks.get(&key)
            };

            match chunk {
                Some(chunk) => {
                    if let Some(payload_id) =
                        self.intern_cached_chunk_payload(build_cached_chunk_payload(chunk))
                    {
                        self.voxel_chunk_payload_cache
                            .insert(key, ChunkPayloadCacheEntry { payload_id });
                        self.add_active_chunk(key);
                    } else {
                        self.remove_active_chunk(key);
                    }
                }
                _ => {
                    self.remove_active_chunk(key);
                }
            }
        }

        self.voxel_world_revision = self.voxel_world_revision.wrapping_add(1);
    }

    fn rebuild_visible_voxel_metadata(
        &mut self,
        cam_pos: [f32; 4],
        cam_forward: [f32; 4],
        near_lod_max_distance: f32,
        mid_lod_max_distance: f32,
        max_trace_distance: f32,
    ) {
        struct YSliceBuildData {
            min_chunk_x: i32,
            max_chunk_x: i32,
            min_chunk_z: i32,
            max_chunk_z: i32,
            min_chunk_w: i32,
            max_chunk_w: i32,
            chunk_coords_xzw: Vec<([i32; 3], u32)>,
        }

        #[derive(Copy, Clone)]
        struct Candidate {
            key: RuntimeChunkKey,
            payload_id: u32,
            min_dist_sq: f32,
        }

        let mut chunk_headers = Vec::new();
        let mut visible_chunk_indices = Vec::new();
        let mut y_slice_build: BTreeMap<i32, YSliceBuildData> = BTreeMap::new();
        let mut y_slice_lookup_entries = Vec::new();

        let trace_max_distance = max_trace_distance.max(1.0);
        let near_max_distance = near_lod_max_distance.clamp(1.0, trace_max_distance);
        let mid_max_distance = mid_lod_max_distance.clamp(near_max_distance, trace_max_distance);
        let near_max_sq = near_max_distance * near_max_distance;
        let mid_max_sq = mid_max_distance * mid_max_distance;
        let trace_max_sq = trace_max_distance * trace_max_distance;
        let view_forward = Self::normalize4_with_fallback(cam_forward, [0.0, 0.0, 1.0, 0.0]);

        let mut candidates = Vec::<Candidate>::with_capacity(self.voxel_active_chunks.len());
        for &key in &self.voxel_active_chunks {
            if key.lod_level != VOXEL_LOD_LEVEL_NEAR
                && key.lod_level != VOXEL_LOD_LEVEL_MID
                && key.lod_level != VOXEL_LOD_LEVEL_FAR
            {
                continue;
            }

            // Cheap directional cull: drop chunks fully behind the camera's
            // current forward half-space so Stage A traces fewer chunk candidates.
            let lod_scale = Self::lod_chunk_world_scale(key.lod_level);
            let chunk_span = (CHUNK_SIZE as f32) * lod_scale;
            let half_span = 0.5 * chunk_span;
            let center = [
                (key.chunk_pos.x as f32 + 0.5) * chunk_span,
                (key.chunk_pos.y as f32 + 0.5) * chunk_span,
                (key.chunk_pos.z as f32 + 0.5) * chunk_span,
                (key.chunk_pos.w as f32 + 0.5) * chunk_span,
            ];
            let to_center = [
                center[0] - cam_pos[0],
                center[1] - cam_pos[1],
                center[2] - cam_pos[2],
                center[3] - cam_pos[3],
            ];
            let forward_center = Self::dot4(to_center, view_forward);
            let forward_radius = half_span
                * (view_forward[0].abs()
                    + view_forward[1].abs()
                    + view_forward[2].abs()
                    + view_forward[3].abs());
            if forward_center + forward_radius < 0.0 {
                continue;
            }

            let Some(chunk_payload_mapping) = self.voxel_chunk_payload_cache.get(&key) else {
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
            let (min_dist_sq, max_dist_sq) = Self::chunk_distance_bounds_sq(cam_pos, key);
            let in_ring = match key.lod_level {
                VOXEL_LOD_LEVEL_NEAR => {
                    min_dist_sq <= near_max_sq
                }
                VOXEL_LOD_LEVEL_MID => {
                    min_dist_sq <= mid_max_sq && max_dist_sq >= near_max_sq
                }
                VOXEL_LOD_LEVEL_FAR => {
                    min_dist_sq <= trace_max_sq && max_dist_sq >= mid_max_sq
                }
                _ => false,
            };
            if !in_ring {
                continue;
            }

            candidates.push(Candidate {
                key,
                payload_id: chunk_payload_mapping.payload_id,
                min_dist_sq,
            });
        }

        candidates.sort_unstable_by(|a, b| {
            a.min_dist_sq
                .total_cmp(&b.min_dist_sq)
                .then_with(|| a.key.lod_level.cmp(&b.key.lod_level))
                .then_with(|| a.key.chunk_pos.w.cmp(&b.key.chunk_pos.w))
                .then_with(|| a.key.chunk_pos.z.cmp(&b.key.chunk_pos.z))
                .then_with(|| a.key.chunk_pos.y.cmp(&b.key.chunk_pos.y))
                .then_with(|| a.key.chunk_pos.x.cmp(&b.key.chunk_pos.x))
        });
        if candidates.len() > VTE_MAX_CHUNKS {
            candidates.truncate(VTE_MAX_CHUNKS);
        }

        for candidate in candidates {
            let key = candidate.key;
            let chunk_pos = key.chunk_pos;
            let payload_idx = candidate.payload_id as usize;
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
                lod_level: key.lod_level as u32,
                _lod_padding: [0; 3],
                occupancy_word_offset,
                material_word_offset,
                flags,
                macro_word_offset,
                solid_local_min: chunk_payload.solid_local_min,
                solid_local_max: chunk_payload.solid_local_max,
            });
            visible_chunk_indices.push(chunk_index);

            if key.lod_level == VOXEL_LOD_LEVEL_NEAR {
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
                        chunk_coords_xzw: vec![(
                            [chunk_pos.x, chunk_pos.z, chunk_pos.w],
                            chunk_index,
                        )],
                    });
            }
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

    /// Build the voxel-native frame payload for VTE.
    pub fn build_voxel_frame_data(
        &mut self,
        cam_pos: [f32; 4],
        cam_forward: [f32; 4],
        near_lod_max_distance: f32,
        mid_lod_max_distance: f32,
        max_trace_distance: f32,
    ) -> &VoxelFrameData {
        self.sync_active_chunk_window(cam_pos, near_lod_max_distance);
        self.process_queued_voxel_payload_updates();

        let cam_chunk = Self::camera_chunk_key(cam_pos);
        let cam_mid_chunk = Self::camera_chunk_key_for_scale(cam_pos, 2);
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
            cam_mid_chunk[0],
            cam_mid_chunk[1],
            cam_mid_chunk[2],
            cam_mid_chunk[3],
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
            && self.voxel_cached_visibility_world_revision == self.voxel_world_revision;
        if !visibility_cache_valid {
            self.rebuild_visible_voxel_metadata(
                cam_pos,
                view_forward,
                near_lod_max_distance,
                mid_lod_max_distance,
                max_trace_distance,
            );
            self.voxel_cached_visibility_camera_chunk = Some(cam_key);
            self.voxel_cached_visibility_world_revision = self.voxel_world_revision;
        }

        self.rebuild_pending_payload_upload_words();
        &self.voxel_frame_data
    }
}
