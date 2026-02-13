use std::collections::HashMap;
use std::io::{self, Read, Write};

pub const CHUNK_SIZE: usize = 8;
pub const CHUNK_VOLUME: usize = CHUNK_SIZE * CHUNK_SIZE * CHUNK_SIZE * CHUNK_SIZE;

const MAGIC: &[u8; 4] = b"V4DW";
const VERSION: u32 = 1;

const ENCODING_UNIFORM: u8 = 0;
const ENCODING_RLE: u8 = 1;
const ENCODING_RAW: u8 = 2;

#[derive(Copy, Clone, Debug, PartialEq, Eq)]
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

#[derive(Debug)]
pub struct Chunk {
    pub voxels: Box<[VoxelType; CHUNK_VOLUME]>,
    pub solid_count: u32,
}

impl Chunk {
    pub fn new() -> Self {
        Self {
            voxels: Box::new([VoxelType::AIR; CHUNK_VOLUME]),
            solid_count: 0,
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
        }
    }

    pub fn is_empty(&self) -> bool {
        self.solid_count == 0
    }
}

#[derive(Debug)]
pub struct VoxelWorld {
    pub chunks: HashMap<ChunkPos, Chunk>,
}

impl VoxelWorld {
    pub fn new() -> Self {
        Self {
            chunks: HashMap::new(),
        }
    }

    pub fn set_voxel(&mut self, wx: i32, wy: i32, wz: i32, ww: i32, voxel: VoxelType) {
        let (chunk_pos, local_index) = world_to_chunk(wx, wy, wz, ww);
        let chunk = self.chunks.entry(chunk_pos).or_insert_with(Chunk::new);
        let old = chunk.voxels[local_index];
        if old == voxel {
            return;
        }
        if old.is_solid() && voxel.is_air() {
            chunk.solid_count -= 1;
        } else if old.is_air() && voxel.is_solid() {
            chunk.solid_count += 1;
        }
        chunk.voxels[local_index] = voxel;
    }

    pub fn insert_chunk(&mut self, pos: ChunkPos, chunk: Chunk) {
        self.chunks.insert(pos, chunk);
    }

    pub fn non_empty_chunk_count(&self) -> usize {
        self.chunks.values().filter(|c| !c.is_empty()).count()
    }
}

fn world_to_chunk(wx: i32, wy: i32, wz: i32, ww: i32) -> (ChunkPos, usize) {
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

pub fn save_world<W: Write>(world: &VoxelWorld, writer: &mut W) -> io::Result<()> {
    writer.write_all(MAGIC)?;
    writer.write_all(&VERSION.to_le_bytes())?;

    let non_empty: Vec<_> = world.chunks.iter().filter(|(_, c)| !c.is_empty()).collect();
    writer.write_all(&(non_empty.len() as u32).to_le_bytes())?;

    for (&pos, chunk) in &non_empty {
        writer.write_all(&pos.x.to_le_bytes())?;
        writer.write_all(&pos.y.to_le_bytes())?;
        writer.write_all(&pos.z.to_le_bytes())?;
        writer.write_all(&pos.w.to_le_bytes())?;
        write_chunk(chunk, writer)?;
    }

    Ok(())
}

pub fn load_world<R: Read>(reader: &mut R) -> io::Result<VoxelWorld> {
    let mut magic = [0u8; 4];
    reader.read_exact(&mut magic)?;
    if &magic != MAGIC {
        return Err(io::Error::new(io::ErrorKind::InvalidData, "invalid magic"));
    }

    let mut buf4 = [0u8; 4];
    reader.read_exact(&mut buf4)?;
    let version = u32::from_le_bytes(buf4);
    if version != VERSION {
        return Err(io::Error::new(
            io::ErrorKind::InvalidData,
            format!("unsupported version {version}"),
        ));
    }

    reader.read_exact(&mut buf4)?;
    let chunk_count = u32::from_le_bytes(buf4) as usize;

    let mut world = VoxelWorld::new();
    for _ in 0..chunk_count {
        reader.read_exact(&mut buf4)?;
        let x = i32::from_le_bytes(buf4);
        reader.read_exact(&mut buf4)?;
        let y = i32::from_le_bytes(buf4);
        reader.read_exact(&mut buf4)?;
        let z = i32::from_le_bytes(buf4);
        reader.read_exact(&mut buf4)?;
        let w = i32::from_le_bytes(buf4);

        let chunk = read_chunk(reader)?;
        world.insert_chunk(ChunkPos::new(x, y, z, w), chunk);
    }

    Ok(world)
}

fn write_chunk<W: Write>(chunk: &Chunk, writer: &mut W) -> io::Result<()> {
    let first = chunk.voxels[0];
    if chunk.voxels.iter().all(|&v| v == first) {
        writer.write_all(&[ENCODING_UNIFORM])?;
        writer.write_all(&[first.0])?;
        return Ok(());
    }

    let runs = rle_encode(&chunk.voxels);
    let rle_size = 1 + 2 + runs.len() * 3;
    let raw_size = 1 + CHUNK_VOLUME;

    if rle_size < raw_size {
        writer.write_all(&[ENCODING_RLE])?;
        writer.write_all(&(runs.len() as u16).to_le_bytes())?;
        for (voxel, length) in runs {
            writer.write_all(&[voxel.0])?;
            writer.write_all(&(length as u16).to_le_bytes())?;
        }
    } else {
        writer.write_all(&[ENCODING_RAW])?;
        let bytes: Vec<u8> = chunk.voxels.iter().map(|v| v.0).collect();
        writer.write_all(&bytes)?;
    }

    Ok(())
}

fn read_chunk<R: Read>(reader: &mut R) -> io::Result<Chunk> {
    let mut enc = [0u8; 1];
    reader.read_exact(&mut enc)?;

    match enc[0] {
        ENCODING_UNIFORM => {
            let mut val = [0u8; 1];
            reader.read_exact(&mut val)?;
            Ok(Chunk::new_filled(VoxelType(val[0])))
        }
        ENCODING_RLE => {
            let mut buf2 = [0u8; 2];
            reader.read_exact(&mut buf2)?;
            let run_count = u16::from_le_bytes(buf2) as usize;

            let mut voxels = Box::new([VoxelType::AIR; CHUNK_VOLUME]);
            let mut pos = 0usize;
            let mut solid_count = 0u32;

            for _ in 0..run_count {
                let mut val = [0u8; 1];
                reader.read_exact(&mut val)?;
                reader.read_exact(&mut buf2)?;
                let length = u16::from_le_bytes(buf2) as usize;
                let voxel = VoxelType(val[0]);

                if pos + length > CHUNK_VOLUME {
                    return Err(io::Error::new(io::ErrorKind::InvalidData, "RLE overflow"));
                }

                for v in &mut voxels[pos..pos + length] {
                    *v = voxel;
                }
                if voxel.is_solid() {
                    solid_count += length as u32;
                }
                pos += length;
            }

            if pos != CHUNK_VOLUME {
                return Err(io::Error::new(
                    io::ErrorKind::InvalidData,
                    "RLE did not fill chunk",
                ));
            }

            Ok(Chunk {
                voxels,
                solid_count,
            })
        }
        ENCODING_RAW => {
            let mut bytes = vec![0u8; CHUNK_VOLUME];
            reader.read_exact(&mut bytes)?;
            let mut voxels = Box::new([VoxelType::AIR; CHUNK_VOLUME]);
            let mut solid_count = 0u32;
            for (i, b) in bytes.into_iter().enumerate() {
                voxels[i] = VoxelType(b);
                if b != 0 {
                    solid_count += 1;
                }
            }
            Ok(Chunk {
                voxels,
                solid_count,
            })
        }
        _ => Err(io::Error::new(
            io::ErrorKind::InvalidData,
            format!("unknown encoding {}", enc[0]),
        )),
    }
}

fn rle_encode(voxels: &[VoxelType; CHUNK_VOLUME]) -> Vec<(VoxelType, usize)> {
    let mut runs = Vec::new();
    let mut current = voxels[0];
    let mut length = 1usize;

    for &v in &voxels[1..] {
        if v == current && length < u16::MAX as usize {
            length += 1;
        } else {
            runs.push((current, length));
            current = v;
            length = 1;
        }
    }
    runs.push((current, length));
    runs
}
