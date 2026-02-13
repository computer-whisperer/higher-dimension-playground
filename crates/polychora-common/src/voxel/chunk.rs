use super::{VoxelType, CHUNK_SIZE, CHUNK_VOLUME};

#[derive(Clone, Debug)]
pub struct Chunk {
    pub voxels: Box<[VoxelType; CHUNK_VOLUME]>,
    pub solid_count: u32,
    pub dirty: bool,
}

impl Chunk {
    pub fn new() -> Self {
        Self {
            voxels: Box::new([VoxelType::AIR; CHUNK_VOLUME]),
            solid_count: 0,
            dirty: true,
        }
    }

    pub fn new_filled(voxel: VoxelType) -> Self {
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

    pub fn get(&self, x: usize, y: usize, z: usize, w: usize) -> VoxelType {
        self.voxels[Self::local_index(x, y, z, w)]
    }

    pub fn set(&mut self, x: usize, y: usize, z: usize, w: usize, v: VoxelType) {
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

    pub fn is_full(&self) -> bool {
        self.solid_count == CHUNK_VOLUME as u32
    }
}
