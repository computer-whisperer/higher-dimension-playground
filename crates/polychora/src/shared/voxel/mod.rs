pub mod chunk;
pub mod io;
pub mod world;

pub use chunk::Chunk;
pub use io::{load_world, save_world};
pub use world::{BaseWorldKind, VoxelWorld};

pub const CHUNK_SIZE: usize = 8;
pub const CHUNK_VOLUME: usize = CHUNK_SIZE * CHUNK_SIZE * CHUNK_SIZE * CHUNK_SIZE; // 4096

#[derive(Copy, Clone, Debug, PartialEq, Eq, Default)]
pub struct VoxelType(pub u8);

impl VoxelType {
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

#[derive(Copy, Clone, Debug, PartialEq, Eq, Hash)]
pub struct ChunkPos {
    pub x: i32,
    pub y: i32,
    pub z: i32,
    pub w: i32,
}

impl ChunkPos {
    pub fn new(x: i32, y: i32, z: i32, w: i32) -> Self {
        Self { x, y, z, w }
    }
}

/// Convert world coords to (chunk_pos, local_index).
pub fn world_to_chunk(wx: i32, wy: i32, wz: i32, ww: i32) -> (ChunkPos, usize) {
    let cs = CHUNK_SIZE as i32;
    let chunk = ChunkPos::new(
        wx.div_euclid(cs),
        wy.div_euclid(cs),
        wz.div_euclid(cs),
        ww.div_euclid(cs),
    );
    let lx = wx.rem_euclid(cs) as usize;
    let ly = wy.rem_euclid(cs) as usize;
    let lz = wz.rem_euclid(cs) as usize;
    let lw = ww.rem_euclid(cs) as usize;
    let idx = lw * CHUNK_SIZE * CHUNK_SIZE * CHUNK_SIZE
        + lz * CHUNK_SIZE * CHUNK_SIZE
        + ly * CHUNK_SIZE
        + lx;
    (chunk, idx)
}
