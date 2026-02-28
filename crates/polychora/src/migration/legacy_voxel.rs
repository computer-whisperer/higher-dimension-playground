use std::collections::{HashMap, HashSet};

use crate::shared::voxel::{
    world_to_chunk_at_scale, BaseWorldKind, CHUNK_SIZE, CHUNK_VOLUME,
};

/// Legacy chunk position used only during migration from old save formats.
/// Replaces the removed `ChunkPos` struct.
type LegacyChunkPos = [i32; 4];

fn chunk_key_to_legacy(key: crate::shared::region_tree::ChunkKey) -> LegacyChunkPos {
    [
        key[0].to_num::<i32>(),
        key[1].to_num::<i32>(),
        key[2].to_num::<i32>(),
        key[3].to_num::<i32>(),
    ]
}

/// Legacy voxel type used by the old save format (v1/v2 .v4dw files).
/// Wraps a raw u8 material ID; only used during migration.
#[derive(Copy, Clone, Debug, PartialEq, Eq, Default)]
pub struct LegacyVoxel(pub u8);

impl LegacyVoxel {
    pub const AIR: Self = Self(0);
    #[inline]
    pub fn is_air(self) -> bool {
        self.0 == 0
    }
    #[inline]
    pub fn is_solid(self) -> bool {
        self.0 != 0
    }
}

const FLAT_FLOOR_CHUNK_Y: i32 = -1;

#[derive(Clone, Debug)]
pub struct Chunk {
    pub voxels: Box<[LegacyVoxel; CHUNK_VOLUME]>,
    pub solid_count: u32,
    pub dirty: bool,
}

impl Chunk {
    pub fn new() -> Self {
        Self {
            voxels: Box::new([LegacyVoxel::AIR; CHUNK_VOLUME]),
            solid_count: 0,
            dirty: true,
        }
    }

    pub fn new_filled(voxel: LegacyVoxel) -> Self {
        let solid_count = if voxel.is_solid() {
            CHUNK_VOLUME as u32
        } else {
            0
        };
        Self {
            voxels: Box::new([voxel; CHUNK_VOLUME]),
            solid_count,
            dirty: true,
        }
    }

    #[inline]
    pub fn local_index(x: usize, y: usize, z: usize, w: usize) -> usize {
        debug_assert!(x < CHUNK_SIZE && y < CHUNK_SIZE && z < CHUNK_SIZE && w < CHUNK_SIZE);
        w * CHUNK_SIZE * CHUNK_SIZE * CHUNK_SIZE + z * CHUNK_SIZE * CHUNK_SIZE + y * CHUNK_SIZE + x
    }

    pub fn get(&self, x: usize, y: usize, z: usize, w: usize) -> LegacyVoxel {
        self.voxels[Self::local_index(x, y, z, w)]
    }

    pub fn set(&mut self, x: usize, y: usize, z: usize, w: usize, v: LegacyVoxel) {
        let idx = Self::local_index(x, y, z, w);
        let old = self.voxels[idx];
        if old == v {
            return;
        }
        if old.is_solid() && v.is_air() {
            self.solid_count -= 1;
        } else if old.is_air() && v.is_solid() {
            self.solid_count += 1;
        }
        self.voxels[idx] = v;
        self.dirty = true;
    }

    pub fn is_empty(&self) -> bool {
        self.solid_count == 0
    }
}

#[derive(Debug)]
pub struct RegionChunkWorld {
    pub chunks: HashMap<LegacyChunkPos, Chunk>,
    base_kind: BaseWorldKind,
    flat_floor_chunk: Chunk,
    world_dirty: bool,
    pending_chunk_updates: Vec<LegacyChunkPos>,
    pending_chunk_update_set: HashSet<LegacyChunkPos>,
}

impl RegionChunkWorld {
    pub fn new() -> Self {
        Self::new_with_base(BaseWorldKind::Empty)
    }

    pub fn new_with_base(base_kind: BaseWorldKind) -> Self {
        let flat_floor_chunk = match &base_kind {
            BaseWorldKind::FlatFloor { material }
            | BaseWorldKind::MassivePlatforms { material } => {
                Self::build_flat_floor_chunk(LegacyVoxel(crate::content_registry::material_token_from_block_data(material)))
            }
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

    fn queue_chunk_update(&mut self, pos: LegacyChunkPos) {
        if self.pending_chunk_update_set.insert(pos) {
            self.pending_chunk_updates.push(pos);
        }
    }

    fn build_flat_floor_chunk(material: LegacyVoxel) -> Chunk {
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

    fn base_chunk_for_pos(&self, pos: LegacyChunkPos) -> Option<&Chunk> {
        match self.base_kind {
            BaseWorldKind::Empty => None,
            BaseWorldKind::FlatFloor { .. } | BaseWorldKind::MassivePlatforms { .. }
                if pos[1] == FLAT_FLOOR_CHUNK_Y =>
            {
                Some(&self.flat_floor_chunk)
            }
            BaseWorldKind::FlatFloor { .. } | BaseWorldKind::MassivePlatforms { .. } => None,
        }
    }

    fn base_voxel_at(&self, pos: LegacyChunkPos, idx: usize) -> LegacyVoxel {
        self.base_chunk_for_pos(pos)
            .map(|chunk| chunk.voxels[idx])
            .unwrap_or(LegacyVoxel::AIR)
    }

    fn clone_base_chunk_or_empty(&self, pos: LegacyChunkPos) -> Chunk {
        self.base_chunk_for_pos(pos)
            .cloned()
            .unwrap_or_else(Chunk::new)
    }

    fn chunk_matches_base(&self, pos: LegacyChunkPos, chunk: &Chunk) -> bool {
        match self.base_chunk_for_pos(pos) {
            Some(base) => {
                chunk.solid_count == base.solid_count && chunk.voxels[..] == base.voxels[..]
            }
            None => chunk.is_empty(),
        }
    }

    fn set_chunk_voxel(chunk: &mut Chunk, idx: usize, v: LegacyVoxel) {
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
        self.base_kind.clone()
    }

    pub fn insert_chunk(&mut self, pos: LegacyChunkPos, mut chunk: Chunk) {
        chunk.dirty = true;
        if self.chunk_matches_base(pos, &chunk) {
            self.chunks.remove(&pos);
        } else {
            self.chunks.insert(pos, chunk);
        }
        self.world_dirty = true;
        self.queue_chunk_update(pos);
    }

    pub fn set_voxel(&mut self, wx: i32, wy: i32, wz: i32, ww: i32, v: LegacyVoxel) {
        let (ck, idx) = world_to_chunk_at_scale(wx, wy, wz, ww, 0);
        let cp = chunk_key_to_legacy(ck);
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
                let neighbor = [cp[0] + dx, cp[1] + dy, cp[2] + dz, cp[3] + dw];
                if let Some(nc) = self.chunks.get_mut(&neighbor) {
                    nc.dirty = true;
                }
            }
        }
    }

    pub fn get_voxel(&self, wx: i32, wy: i32, wz: i32, ww: i32) -> LegacyVoxel {
        let (ck, idx) = world_to_chunk_at_scale(wx, wy, wz, ww, 0);
        let cp = chunk_key_to_legacy(ck);
        match self.chunks.get(&cp) {
            Some(chunk) => chunk.voxels[idx],
            None => self.base_voxel_at(cp, idx),
        }
    }

    pub fn remove_chunk_override(&mut self, pos: LegacyChunkPos) -> bool {
        let removed = self.chunks.remove(&pos).is_some();
        if removed {
            self.world_dirty = true;
            self.queue_chunk_update(pos);
        }
        removed
    }

    pub fn non_empty_chunk_count(&self) -> usize {
        self.chunks
            .values()
            .filter(|chunk| !chunk.is_empty())
            .count()
    }

    pub fn drain_pending_chunk_updates(&mut self) -> Vec<LegacyChunkPos> {
        self.pending_chunk_update_set.clear();
        std::mem::take(&mut self.pending_chunk_updates)
    }

    pub fn clear_dirty(&mut self) {
        self.world_dirty = false;
        for chunk in self.chunks.values_mut() {
            chunk.dirty = false;
        }
    }
}
