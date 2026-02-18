use crate::shared::voxel::{Chunk, VoxelType, CHUNK_SIZE, CHUNK_VOLUME};
use crate::shared::worldfield::{Aabb4i, ChunkKey, ChunkPayload, RegionChunkTree};
use std::collections::HashMap;

#[derive(Clone, Debug)]
pub struct DeterministicRng {
    state: u64,
}

impl DeterministicRng {
    pub fn new(seed: u64) -> Self {
        let mut state = seed ^ 0x9e37_79b9_7f4a_7c15;
        if state == 0 {
            state = 0x2545_f491_4f6c_dd1d;
        }
        Self { state }
    }

    pub fn next_u64(&mut self) -> u64 {
        let mut x = self.state;
        x ^= x >> 12;
        x ^= x << 25;
        x ^= x >> 27;
        self.state = x;
        x.wrapping_mul(0x2545_f491_4f6c_dd1d)
    }

    pub fn chance(&mut self, numerator: u64, denominator: u64) -> bool {
        if denominator == 0 {
            return false;
        }
        (self.next_u64() % denominator) < numerator
    }

    pub fn range_i32(&mut self, min: i32, max: i32) -> i32 {
        debug_assert!(min <= max);
        let span = (i64::from(max) - i64::from(min) + 1) as u64;
        min + (self.next_u64() % span) as i32
    }

    pub fn range_usize(&mut self, min: usize, max: usize) -> usize {
        debug_assert!(min <= max);
        let span = (max - min) as u64 + 1;
        min + (self.next_u64() % span) as usize
    }

    pub fn range_u16(&mut self, min: u16, max: u16) -> u16 {
        debug_assert!(min <= max);
        let span = u64::from(max - min) + 1;
        min + (self.next_u64() % span) as u16
    }
}

#[derive(Clone, Debug, Default)]
pub struct ReferenceChunkStore {
    chunks: HashMap<ChunkKey, ChunkPayload>,
}

impl ReferenceChunkStore {
    pub fn new() -> Self {
        Self::default()
    }

    pub fn from_tree(tree: &RegionChunkTree) -> Self {
        Self {
            chunks: tree.collect_chunks().into_iter().collect(),
        }
    }

    pub fn set_chunk(&mut self, key: ChunkKey, payload: Option<ChunkPayload>) -> bool {
        match payload {
            Some(payload) => {
                let payload = canonicalize_payload(payload);
                if self.chunks.get(&key) == Some(&payload) {
                    return false;
                }
                self.chunks.insert(key, payload);
                true
            }
            None => self.chunks.remove(&key).is_some(),
        }
    }

    pub fn chunk_payload(&self, key: ChunkKey) -> Option<ChunkPayload> {
        self.chunks.get(&key).cloned()
    }

    pub fn collect_chunks_sorted(&self) -> Vec<(ChunkKey, ChunkPayload)> {
        let mut chunks: Vec<_> = self
            .chunks
            .iter()
            .map(|(key, payload)| (*key, payload.clone()))
            .collect();
        chunks.sort_unstable_by_key(|(key, _)| key.pos);
        chunks
    }

    pub fn collect_chunks_in_bounds_sorted(&self, bounds: Aabb4i) -> Vec<(ChunkKey, ChunkPayload)> {
        if !bounds.is_valid() {
            return Vec::new();
        }
        let mut chunks: Vec<_> = self
            .chunks
            .iter()
            .filter(|(key, _)| bounds.contains_chunk(key.pos))
            .map(|(key, payload)| (*key, payload.clone()))
            .collect();
        chunks.sort_unstable_by_key(|(key, _)| key.pos);
        chunks
    }

    pub fn any_non_empty_chunk_in_bounds(&self, bounds: Aabb4i) -> bool {
        if !bounds.is_valid() {
            return false;
        }
        self.chunks
            .iter()
            .filter(|(key, _)| bounds.contains_chunk(key.pos))
            .any(|(_, payload)| payload_has_solid_material(payload))
    }

    pub fn non_empty_chunk_count(&self) -> usize {
        self.chunks
            .values()
            .filter(|payload| payload_has_solid_material(payload))
            .count()
    }

    pub fn replace_non_empty_chunks_in_bounds<I>(&mut self, bounds: Aabb4i, desired: I)
    where
        I: IntoIterator<Item = (ChunkKey, ChunkPayload)>,
    {
        if !bounds.is_valid() {
            return;
        }

        let to_remove: Vec<ChunkKey> = self
            .chunks
            .keys()
            .filter(|key| bounds.contains_chunk(key.pos))
            .copied()
            .collect();
        for key in to_remove {
            self.chunks.remove(&key);
        }

        for (key, payload) in desired {
            if !bounds.contains_chunk(key.pos) {
                continue;
            }
            let canonical = canonicalize_payload(payload);
            if payload_has_solid_material(&canonical) {
                self.chunks.insert(key, canonical);
            }
        }
    }
}

pub fn assert_tree_matches_reference(tree: &RegionChunkTree, reference: &ReferenceChunkStore) {
    let mut tree_chunks = tree.collect_chunks();
    tree_chunks.sort_unstable_by_key(|(key, _)| key.pos);
    assert_eq!(tree_chunks, reference.collect_chunks_sorted());
    assert_eq!(
        tree.non_empty_chunk_count(),
        reference.non_empty_chunk_count()
    );
}

pub fn canonicalize_payload(payload: ChunkPayload) -> ChunkPayload {
    let payload = match payload {
        ChunkPayload::Empty => ChunkPayload::Uniform(0),
        other => other,
    };
    let Ok(dense) = payload.dense_materials() else {
        return payload;
    };
    if dense.is_empty() {
        return payload;
    }
    let first = dense[0];
    if dense.iter().all(|m| *m == first) {
        ChunkPayload::Uniform(first)
    } else {
        payload
    }
}

pub fn payload_has_solid_material(payload: &ChunkPayload) -> bool {
    match payload {
        ChunkPayload::Empty => false,
        ChunkPayload::Uniform(material) => *material != 0,
        ChunkPayload::Dense16 { materials } => materials.iter().any(|material| *material != 0),
        ChunkPayload::PalettePacked { .. } => payload
            .dense_materials()
            .map(|dense| dense.into_iter().any(|material| material != 0))
            .unwrap_or(true),
    }
}

pub fn expand_bounds(bounds: Aabb4i, padding: i32) -> Aabb4i {
    Aabb4i::new(
        [
            bounds.min[0].saturating_sub(padding),
            bounds.min[1].saturating_sub(padding),
            bounds.min[2].saturating_sub(padding),
            bounds.min[3].saturating_sub(padding),
        ],
        [
            bounds.max[0].saturating_add(padding),
            bounds.max[1].saturating_add(padding),
            bounds.max[2].saturating_add(padding),
            bounds.max[3].saturating_add(padding),
        ],
    )
}

pub fn random_chunk_key_in_bounds(rng: &mut DeterministicRng, bounds: Aabb4i) -> ChunkKey {
    ChunkKey {
        pos: [
            rng.range_i32(bounds.min[0], bounds.max[0]),
            rng.range_i32(bounds.min[1], bounds.max[1]),
            rng.range_i32(bounds.min[2], bounds.max[2]),
            rng.range_i32(bounds.min[3], bounds.max[3]),
        ],
    }
}

pub fn random_sub_bounds(rng: &mut DeterministicRng, bounds: Aabb4i) -> Aabb4i {
    if !bounds.is_valid() {
        return bounds;
    }
    let mut min = [0i32; 4];
    let mut max = [0i32; 4];
    for axis in 0..4 {
        let a = rng.range_i32(bounds.min[axis], bounds.max[axis]);
        let b = rng.range_i32(bounds.min[axis], bounds.max[axis]);
        min[axis] = a.min(b);
        max[axis] = a.max(b);
    }
    Aabb4i::new(min, max)
}

pub fn random_chunk_payload(rng: &mut DeterministicRng) -> ChunkPayload {
    match rng.range_usize(0, 99) {
        0..=39 => ChunkPayload::Uniform(rng.range_u16(0, 31)),
        40..=54 => ChunkPayload::Empty,
        55..=84 => random_dense_payload(rng),
        _ => random_palette_payload(rng),
    }
}

fn random_dense_payload(rng: &mut DeterministicRng) -> ChunkPayload {
    let base = rng.range_u16(0, 31);
    let mut materials = vec![base; CHUNK_VOLUME];
    let edits = rng.range_usize(1, 8);
    for _ in 0..edits {
        let idx = rng.range_usize(0, CHUNK_VOLUME - 1);
        materials[idx] = rng.range_u16(0, 31);
    }
    ChunkPayload::Dense16 { materials }
}

fn random_palette_payload(rng: &mut DeterministicRng) -> ChunkPayload {
    let mut chunk = Chunk::new();
    let edits = rng.range_usize(1, 12);
    for _ in 0..edits {
        let index = rng.range_usize(0, CHUNK_VOLUME - 1);
        let coords = local_coords_from_index(index);
        let material = rng.range_u16(0, 31) as u8;
        chunk.set(
            coords[0],
            coords[1],
            coords[2],
            coords[3],
            VoxelType(material),
        );
    }
    ChunkPayload::from_chunk_compact(&chunk)
}

fn local_coords_from_index(index: usize) -> [usize; 4] {
    let mut rem = index;
    let x = rem % CHUNK_SIZE;
    rem /= CHUNK_SIZE;
    let y = rem % CHUNK_SIZE;
    rem /= CHUNK_SIZE;
    let z = rem % CHUNK_SIZE;
    rem /= CHUNK_SIZE;
    let w = rem % CHUNK_SIZE;
    [x, y, z, w]
}
