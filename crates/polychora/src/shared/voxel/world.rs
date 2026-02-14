use std::collections::{HashMap, HashSet};

use super::chunk::Chunk;
use super::{world_to_chunk, ChunkPos, VoxelType, CHUNK_SIZE};

const FLAT_FLOOR_CHUNK_Y: i32 = -1;

#[derive(Copy, Clone, Debug, PartialEq, Eq)]
pub enum BaseWorldKind {
    Empty,
    FlatFloor { material: VoxelType },
}

#[derive(Debug)]
pub struct VoxelWorld {
    /// Sparse full-chunk overrides relative to `base_kind`.
    pub chunks: HashMap<ChunkPos, Chunk>,
    base_kind: BaseWorldKind,
    flat_floor_chunk: Chunk,
    world_dirty: bool,
    pending_chunk_updates: Vec<ChunkPos>,
    pending_chunk_update_set: HashSet<ChunkPos>,
}

impl VoxelWorld {
    pub fn new() -> Self {
        Self::new_with_base(BaseWorldKind::Empty)
    }

    pub fn new_with_base(base_kind: BaseWorldKind) -> Self {
        let flat_floor_chunk = match base_kind {
            BaseWorldKind::FlatFloor { material } => Self::build_flat_floor_chunk(material),
            BaseWorldKind::Empty => Chunk::new(),
        };
        Self {
            chunks: HashMap::new(),
            base_kind,
            flat_floor_chunk,
            world_dirty: false,
            pending_chunk_updates: Vec::new(),
            pending_chunk_update_set: HashSet::new(),
        }
    }

    fn queue_chunk_update(&mut self, pos: ChunkPos) {
        if self.pending_chunk_update_set.insert(pos) {
            self.pending_chunk_updates.push(pos);
        }
    }

    fn build_flat_floor_chunk(material: VoxelType) -> Chunk {
        let mut chunk = Chunk::new();
        if material.is_air() {
            chunk.dirty = false;
            return chunk;
        }

        let local_y_top = CHUNK_SIZE - 1;
        let local_y_bottom = CHUNK_SIZE - 2;
        for x in 0..CHUNK_SIZE {
            for z in 0..CHUNK_SIZE {
                for w in 0..CHUNK_SIZE {
                    chunk.set(x, local_y_top, z, w, material);
                    chunk.set(x, local_y_bottom, z, w, material);
                }
            }
        }
        chunk.dirty = false;
        chunk
    }

    fn base_chunk_for_pos(&self, pos: ChunkPos) -> Option<&Chunk> {
        match self.base_kind {
            BaseWorldKind::Empty => None,
            BaseWorldKind::FlatFloor { .. } if pos.y == FLAT_FLOOR_CHUNK_Y => {
                Some(&self.flat_floor_chunk)
            }
            BaseWorldKind::FlatFloor { .. } => None,
        }
    }

    fn base_voxel_at(&self, pos: ChunkPos, idx: usize) -> VoxelType {
        self.base_chunk_for_pos(pos)
            .map(|chunk| chunk.voxels[idx])
            .unwrap_or(VoxelType::AIR)
    }

    fn clone_base_chunk_or_empty(&self, pos: ChunkPos) -> Chunk {
        self.base_chunk_for_pos(pos)
            .cloned()
            .unwrap_or_else(Chunk::new)
    }

    fn chunk_matches_base(&self, pos: ChunkPos, chunk: &Chunk) -> bool {
        match self.base_chunk_for_pos(pos) {
            Some(base) => {
                chunk.solid_count == base.solid_count && chunk.voxels[..] == base.voxels[..]
            }
            None => chunk.is_empty(),
        }
    }

    fn set_chunk_voxel(chunk: &mut Chunk, idx: usize, v: VoxelType) {
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
    }

    pub fn base_kind(&self) -> BaseWorldKind {
        self.base_kind
    }

    pub fn queue_chunk_refresh(&mut self, pos: ChunkPos) {
        self.queue_chunk_update(pos);
    }

    /// Returns the effective non-empty chunk at `pos`.
    pub fn chunk_at(&self, pos: ChunkPos) -> Option<&Chunk> {
        if let Some(override_chunk) = self.chunks.get(&pos) {
            return (!override_chunk.is_empty()).then_some(override_chunk);
        }
        self.base_chunk_for_pos(pos)
            .filter(|chunk| !chunk.is_empty())
    }

    /// Collect all effective non-empty chunk positions within inclusive chunk bounds.
    pub fn gather_non_empty_chunks_in_bounds(
        &self,
        min_chunk: [i32; 4],
        max_chunk: [i32; 4],
        out: &mut Vec<ChunkPos>,
    ) {
        out.clear();
        if min_chunk[0] > max_chunk[0]
            || min_chunk[1] > max_chunk[1]
            || min_chunk[2] > max_chunk[2]
            || min_chunk[3] > max_chunk[3]
        {
            return;
        }

        if let BaseWorldKind::FlatFloor { .. } = self.base_kind {
            let y = FLAT_FLOOR_CHUNK_Y;
            if y >= min_chunk[1] && y <= max_chunk[1] {
                for x in min_chunk[0]..=max_chunk[0] {
                    for z in min_chunk[2]..=max_chunk[2] {
                        for w in min_chunk[3]..=max_chunk[3] {
                            let pos = ChunkPos::new(x, y, z, w);
                            if let Some(override_chunk) = self.chunks.get(&pos) {
                                if !override_chunk.is_empty() {
                                    out.push(pos);
                                }
                            } else {
                                out.push(pos);
                            }
                        }
                    }
                }
            }
        }

        for (&pos, chunk) in &self.chunks {
            if pos.x < min_chunk[0]
                || pos.x > max_chunk[0]
                || pos.y < min_chunk[1]
                || pos.y > max_chunk[1]
                || pos.z < min_chunk[2]
                || pos.z > max_chunk[2]
                || pos.w < min_chunk[3]
                || pos.w > max_chunk[3]
                || chunk.is_empty()
            {
                continue;
            }
            if matches!(self.base_kind, BaseWorldKind::FlatFloor { .. })
                && pos.y == FLAT_FLOOR_CHUNK_Y
            {
                continue;
            }
            out.push(pos);
        }
    }

    pub fn get_voxel(&self, wx: i32, wy: i32, wz: i32, ww: i32) -> VoxelType {
        let (cp, idx) = world_to_chunk(wx, wy, wz, ww);
        match self.chunks.get(&cp) {
            Some(chunk) => chunk.voxels[idx],
            None => self.base_voxel_at(cp, idx),
        }
    }

    pub fn set_voxel(&mut self, wx: i32, wy: i32, wz: i32, ww: i32, v: VoxelType) {
        let (cp, idx) = world_to_chunk(wx, wy, wz, ww);
        if self.chunks.contains_key(&cp) {
            {
                let chunk = self.chunks.get_mut(&cp).expect("chunk just checked");
                if chunk.voxels[idx] == v {
                    return;
                }
                Self::set_chunk_voxel(chunk, idx, v);
            }
            let remove_override = self
                .chunks
                .get(&cp)
                .map(|updated_chunk| self.chunk_matches_base(cp, updated_chunk))
                .unwrap_or(false);
            if remove_override {
                self.chunks.remove(&cp);
            }
        } else {
            let old = self.base_voxel_at(cp, idx);
            if old == v {
                return;
            }
            let mut chunk = self.clone_base_chunk_or_empty(cp);
            Self::set_chunk_voxel(&mut chunk, idx, v);
            if !self.chunk_matches_base(cp, &chunk) {
                self.chunks.insert(cp, chunk);
            }
        }

        self.world_dirty = true;
        self.queue_chunk_update(cp);

        // Mark override neighbors dirty at boundaries so tetra-surface mode rebuilds.
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
    pub fn insert_chunk(&mut self, pos: ChunkPos, mut chunk: Chunk) {
        chunk.dirty = true;
        if self.chunk_matches_base(pos, &chunk) {
            self.chunks.remove(&pos);
        } else {
            self.chunks.insert(pos, chunk);
        }
        self.world_dirty = true;
        self.queue_chunk_update(pos);
    }

    pub fn non_empty_chunk_count(&self) -> usize {
        self.chunks
            .values()
            .filter(|chunk| !chunk.is_empty())
            .count()
    }

    pub fn drain_pending_chunk_updates(&mut self) -> Vec<ChunkPos> {
        self.pending_chunk_update_set.clear();
        std::mem::take(&mut self.pending_chunk_updates)
    }

    pub fn any_dirty(&self) -> bool {
        self.world_dirty
    }

    pub fn clear_dirty(&mut self) {
        self.world_dirty = false;
        for chunk in self.chunks.values_mut() {
            chunk.dirty = false;
        }
    }
}
