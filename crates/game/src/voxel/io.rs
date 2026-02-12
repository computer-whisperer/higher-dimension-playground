use std::io::{self, Read, Write};

use super::chunk::Chunk;
use super::world::VoxelWorld;
use super::{ChunkPos, VoxelType, CHUNK_VOLUME};

const MAGIC: &[u8; 4] = b"V4DW";
const VERSION: u32 = 1;

const ENCODING_UNIFORM: u8 = 0;
const ENCODING_RLE: u8 = 1;
const ENCODING_RAW: u8 = 2;

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

fn write_chunk<W: Write>(chunk: &Chunk, writer: &mut W) -> io::Result<()> {
    // Check if uniform
    let first = chunk.voxels[0];
    if chunk.voxels.iter().all(|&v| v == first) {
        writer.write_all(&[ENCODING_UNIFORM])?;
        writer.write_all(&[first.0])?;
        return Ok(());
    }

    // Try RLE
    let runs = rle_encode(&chunk.voxels);
    let rle_size = 1 + 2 + runs.len() * 3; // encoding + run_count + runs
    let raw_size = 1 + CHUNK_VOLUME; // encoding + data

    if rle_size < raw_size {
        writer.write_all(&[ENCODING_RLE])?;
        writer.write_all(&(runs.len() as u16).to_le_bytes())?;
        for (voxel, length) in &runs {
            writer.write_all(&[voxel.0])?;
            writer.write_all(&(*length as u16).to_le_bytes())?;
        }
    } else {
        writer.write_all(&[ENCODING_RAW])?;
        let bytes: Vec<u8> = chunk.voxels.iter().map(|v| v.0).collect();
        writer.write_all(&bytes)?;
    }

    Ok(())
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
            let mut pos = 0;
            let mut solid_count = 0u32;

            for _ in 0..run_count {
                let mut val = [0u8; 1];
                reader.read_exact(&mut val)?;
                reader.read_exact(&mut buf2)?;
                let length = u16::from_le_bytes(buf2) as usize;
                let vt = VoxelType(val[0]);

                if pos + length > CHUNK_VOLUME {
                    return Err(io::Error::new(io::ErrorKind::InvalidData, "RLE overflow"));
                }

                for v in &mut voxels[pos..pos + length] {
                    *v = vt;
                }
                if vt.is_solid() {
                    solid_count += length as u32;
                }
                pos += length;
            }

            Ok(Chunk {
                voxels,
                solid_count,
                dirty: true,
            })
        }
        ENCODING_RAW => {
            let mut bytes = vec![0u8; CHUNK_VOLUME];
            reader.read_exact(&mut bytes)?;
            let mut voxels = Box::new([VoxelType::AIR; CHUNK_VOLUME]);
            let mut solid_count = 0u32;
            for (i, &b) in bytes.iter().enumerate() {
                voxels[i] = VoxelType(b);
                if b != 0 {
                    solid_count += 1;
                }
            }
            Ok(Chunk {
                voxels,
                solid_count,
                dirty: true,
            })
        }
        _ => Err(io::Error::new(
            io::ErrorKind::InvalidData,
            format!("unknown encoding {}", enc[0]),
        )),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn round_trip_empty_world() {
        let world = VoxelWorld::new();
        let mut buf = Vec::new();
        save_world(&world, &mut buf).unwrap();
        let loaded = load_world(&mut &buf[..]).unwrap();
        assert_eq!(loaded.chunks.len(), 0);
    }

    #[test]
    fn round_trip_uniform_chunk() {
        let mut world = VoxelWorld::new();
        let chunk = Chunk::new_filled(VoxelType(3));
        world.insert_chunk(ChunkPos::new(1, -2, 3, 0), chunk);

        let mut buf = Vec::new();
        save_world(&world, &mut buf).unwrap();
        let loaded = load_world(&mut &buf[..]).unwrap();

        assert_eq!(loaded.chunks.len(), 1);
        let c = loaded.chunks.get(&ChunkPos::new(1, -2, 3, 0)).unwrap();
        assert!(c.voxels.iter().all(|v| v.0 == 3));
    }

    #[test]
    fn round_trip_mixed_chunk() {
        let mut world = VoxelWorld::new();
        let mut chunk = Chunk::new();
        chunk.set(0, 0, 0, 0, VoxelType(1));
        chunk.set(7, 7, 7, 7, VoxelType(5));
        chunk.set(3, 2, 1, 4, VoxelType(10));
        world.insert_chunk(ChunkPos::new(0, 0, 0, 0), chunk);

        let mut buf = Vec::new();
        save_world(&world, &mut buf).unwrap();
        let loaded = load_world(&mut &buf[..]).unwrap();

        let c = loaded.chunks.get(&ChunkPos::new(0, 0, 0, 0)).unwrap();
        assert_eq!(c.get(0, 0, 0, 0).0, 1);
        assert_eq!(c.get(7, 7, 7, 7).0, 5);
        assert_eq!(c.get(3, 2, 1, 4).0, 10);
        assert_eq!(c.get(1, 1, 1, 1).0, 0); // air
    }
}
