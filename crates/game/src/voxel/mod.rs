pub mod chunk;
pub mod cull;
pub mod io;
pub mod mesh;
pub mod world;
pub mod worldgen;

pub const CHUNK_SIZE: usize = 8;
pub const CHUNK_VOLUME: usize = CHUNK_SIZE * CHUNK_SIZE * CHUNK_SIZE * CHUNK_SIZE; // 4096

#[derive(Copy, Clone, PartialEq, Eq, Default)]
pub struct VoxelType(pub u8);

impl VoxelType {
    pub const AIR: Self = Self(0);

    pub fn is_air(self) -> bool {
        self.0 == 0
    }

    pub fn is_solid(self) -> bool {
        self.0 != 0
    }
}

/// The 8 cubic cells (3-faces) of a tesseract, with their cell IDs matching
/// the order from `Hypercube::<4>::generate_k_faces_3()`.
#[derive(Copy, Clone, Debug)]
#[repr(u8)]
pub enum Face4D {
    NegW = 0,
    NegZ = 1,
    NegY = 2,
    NegX = 3,
    PosW = 4,
    PosZ = 5,
    PosY = 6,
    PosX = 7,
}

impl Face4D {
    pub const ALL: [Face4D; 8] = [
        Face4D::NegW,
        Face4D::NegZ,
        Face4D::NegY,
        Face4D::NegX,
        Face4D::PosW,
        Face4D::PosZ,
        Face4D::PosY,
        Face4D::PosX,
    ];

    pub fn cell_id(self) -> usize {
        self as usize
    }

    /// Offset to the neighboring voxel through this face.
    pub fn neighbor_offset(self) -> [i32; 4] {
        match self {
            Face4D::NegX => [-1, 0, 0, 0],
            Face4D::PosX => [1, 0, 0, 0],
            Face4D::NegY => [0, -1, 0, 0],
            Face4D::PosY => [0, 1, 0, 0],
            Face4D::NegZ => [0, 0, -1, 0],
            Face4D::PosZ => [0, 0, 1, 0],
            Face4D::NegW => [0, 0, 0, -1],
            Face4D::PosW => [0, 0, 0, 1],
        }
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
