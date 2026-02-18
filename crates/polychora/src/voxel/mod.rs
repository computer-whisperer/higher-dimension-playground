pub mod cull;
pub mod world;

pub use polychora::shared::voxel::{ChunkPos, VoxelType, CHUNK_SIZE, CHUNK_VOLUME};

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
