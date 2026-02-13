use std::collections::{HashMap, HashSet};

use super::chunk::Chunk;
use super::{world_to_chunk, ChunkPos, VoxelType, CHUNK_SIZE};

pub struct VoxelWorld {
    pub chunks: HashMap<ChunkPos, Chunk>,
    pending_chunk_updates: Vec<ChunkPos>,
    pending_chunk_update_set: HashSet<ChunkPos>,
}

impl VoxelWorld {
    pub fn new() -> Self {
        Self {
            chunks: HashMap::new(),
            pending_chunk_updates: Vec::new(),
            pending_chunk_update_set: HashSet::new(),
        }
    }

    fn queue_chunk_update(&mut self, pos: ChunkPos) {
        if self.pending_chunk_update_set.insert(pos) {
            self.pending_chunk_updates.push(pos);
        }
    }

    pub fn get_voxel(&self, wx: i32, wy: i32, wz: i32, ww: i32) -> VoxelType {
        let (cp, idx) = world_to_chunk(wx, wy, wz, ww);
        match self.chunks.get(&cp) {
            Some(chunk) => chunk.voxels[idx],
            None => VoxelType::AIR,
        }
    }

    pub fn set_voxel(&mut self, wx: i32, wy: i32, wz: i32, ww: i32, v: VoxelType) {
        let (cp, idx) = world_to_chunk(wx, wy, wz, ww);
        let chunk = self.chunks.entry(cp).or_insert_with(Chunk::new);

        let old = chunk.voxels[idx];
        if old == v {
            return;
        }
        if old.is_solid() && v.is_air() {
            chunk.solid_count -= 1;
        } else if old.is_air() && v.is_solid() {
            chunk.solid_count += 1;
        }
        chunk.voxels[idx] = v;
        chunk.dirty = true;
        self.queue_chunk_update(cp);

        // Mark neighbor chunks dirty at boundaries
        let cs = CHUNK_SIZE as i32;
        let lx = wx.rem_euclid(cs);
        let ly = wy.rem_euclid(cs);
        let lz = wz.rem_euclid(cs);
        let lw = ww.rem_euclid(cs);

        let offsets: &[(i32, [i32; 4])] = &[
            (lx, [-1, 0, 0, 0]),
            (cs - 1 - lx, [1, 0, 0, 0]),
            (ly, [0, -1, 0, 0]),
            (cs - 1 - ly, [0, 1, 0, 0]),
            (lz, [0, 0, -1, 0]),
            (cs - 1 - lz, [0, 0, 1, 0]),
            (lw, [0, 0, 0, -1]),
            (cs - 1 - lw, [0, 0, 0, 1]),
        ];

        for &(dist, [dx, dy, dz, dw]) in offsets {
            if dist == 0 {
                let neighbor = ChunkPos::new(cp.x + dx, cp.y + dy, cp.z + dz, cp.w + dw);
                if let Some(nc) = self.chunks.get_mut(&neighbor) {
                    nc.dirty = true;
                }
            }
        }
    }

    /// Insert a pre-built chunk at the given position.
    pub fn insert_chunk(&mut self, pos: ChunkPos, chunk: Chunk) {
        self.chunks.insert(pos, chunk);
        self.queue_chunk_update(pos);
    }

    pub fn drain_pending_chunk_updates(&mut self) -> Vec<ChunkPos> {
        self.pending_chunk_update_set.clear();
        std::mem::take(&mut self.pending_chunk_updates)
    }

    pub fn any_dirty(&self) -> bool {
        self.chunks.values().any(|c| c.dirty)
    }

    pub fn clear_dirty(&mut self) {
        for chunk in self.chunks.values_mut() {
            chunk.dirty = false;
        }
    }
}
