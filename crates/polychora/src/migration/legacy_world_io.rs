use std::collections::HashMap;
use std::io::{self, Read, Write};

use crate::migration::legacy_voxel::{Chunk, RegionChunkWorld};
use crate::shared::voxel::{BaseWorldKind, ChunkPos, VoxelType, CHUNK_VOLUME};

const MAGIC: &[u8; 4] = b"V4DW";
const VERSION_V1: u32 = 1;
const VERSION_V2: u32 = 2;
const VERSION: u32 = VERSION_V2;

const ENCODING_UNIFORM: u8 = 0;
const ENCODING_RLE: u8 = 1;
const ENCODING_RAW: u8 = 2;

const BASE_KIND_EMPTY: u8 = 0;
const BASE_KIND_FLAT_FLOOR: u8 = 1;

pub fn save_world<W: Write>(world: &RegionChunkWorld, writer: &mut W) -> io::Result<()> {
    writer.write_all(MAGIC)?;
    writer.write_all(&VERSION.to_le_bytes())?;

    write_base_kind(world.base_kind(), writer)?;

    let mut overrides: Vec<_> = world
        .chunks
        .iter()
        .map(|(&pos, chunk)| (pos, chunk))
        .collect();
    overrides.sort_unstable_by_key(|(pos, _)| (pos.w, pos.z, pos.y, pos.x));

    let mut payloads: Vec<&Chunk> = Vec::new();
    let mut payload_hash_buckets: HashMap<u64, Vec<u32>> = HashMap::new();
    let mut entries = Vec::with_capacity(overrides.len());

    for (pos, chunk) in overrides {
        let payload_index = intern_payload(chunk, &mut payloads, &mut payload_hash_buckets) as u32;
        entries.push((pos, payload_index));
    }

    writer.write_all(&(entries.len() as u32).to_le_bytes())?;
    writer.write_all(&(payloads.len() as u32).to_le_bytes())?;

    for chunk in payloads {
        write_chunk(chunk, writer)?;
    }

    for (pos, payload_index) in entries {
        writer.write_all(&pos.x.to_le_bytes())?;
        writer.write_all(&pos.y.to_le_bytes())?;
        writer.write_all(&pos.z.to_le_bytes())?;
        writer.write_all(&pos.w.to_le_bytes())?;
        writer.write_all(&payload_index.to_le_bytes())?;
    }

    Ok(())
}

fn write_base_kind<W: Write>(base_kind: BaseWorldKind, writer: &mut W) -> io::Result<()> {
    match base_kind {
        BaseWorldKind::Empty => writer.write_all(&[BASE_KIND_EMPTY])?,
        BaseWorldKind::FlatFloor { material } => {
            writer.write_all(&[BASE_KIND_FLAT_FLOOR])?;
            writer.write_all(&[material.0])?;
        }
    }
    Ok(())
}

fn read_base_kind<R: Read>(reader: &mut R) -> io::Result<BaseWorldKind> {
    let mut tag = [0u8; 1];
    reader.read_exact(&mut tag)?;
    match tag[0] {
        BASE_KIND_EMPTY => Ok(BaseWorldKind::Empty),
        BASE_KIND_FLAT_FLOOR => {
            let mut material = [0u8; 1];
            reader.read_exact(&mut material)?;
            Ok(BaseWorldKind::FlatFloor {
                material: VoxelType(material[0]),
            })
        }
        _ => Err(io::Error::new(
            io::ErrorKind::InvalidData,
            format!("unknown base kind {}", tag[0]),
        )),
    }
}

fn chunk_payload_hash(chunk: &Chunk) -> u64 {
    let mut hash = 0xcbf2_9ce4_8422_2325u64;
    hash ^= chunk.solid_count as u64;
    hash = hash.wrapping_mul(0x0000_0100_0000_01b3);
    for voxel in chunk.voxels.iter() {
        hash ^= voxel.0 as u64;
        hash = hash.wrapping_mul(0x0000_0100_0000_01b3);
    }
    hash
}

fn chunks_equal(a: &Chunk, b: &Chunk) -> bool {
    a.solid_count == b.solid_count && a.voxels[..] == b.voxels[..]
}

fn intern_payload<'a>(
    chunk: &'a Chunk,
    payloads: &mut Vec<&'a Chunk>,
    payload_hash_buckets: &mut HashMap<u64, Vec<u32>>,
) -> usize {
    let hash = chunk_payload_hash(chunk);
    if let Some(candidates) = payload_hash_buckets.get(&hash) {
        for &candidate_index in candidates {
            let candidate = payloads[candidate_index as usize];
            if chunks_equal(candidate, chunk) {
                return candidate_index as usize;
            }
        }
    }

    let new_index = payloads.len() as u32;
    payloads.push(chunk);
    payload_hash_buckets
        .entry(hash)
        .or_default()
        .push(new_index);
    new_index as usize
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

pub fn load_world<R: Read>(reader: &mut R) -> io::Result<RegionChunkWorld> {
    let mut magic = [0u8; 4];
    reader.read_exact(&mut magic)?;
    if &magic != MAGIC {
        return Err(io::Error::new(io::ErrorKind::InvalidData, "invalid magic"));
    }

    let mut buf4 = [0u8; 4];
    reader.read_exact(&mut buf4)?;
    let version = u32::from_le_bytes(buf4);

    match version {
        VERSION_V1 => load_world_v1(reader),
        VERSION_V2 => load_world_v2(reader),
        _ => Err(io::Error::new(
            io::ErrorKind::InvalidData,
            format!("unsupported version {version}"),
        )),
    }
}

fn load_world_v1<R: Read>(reader: &mut R) -> io::Result<RegionChunkWorld> {
    let mut buf4 = [0u8; 4];
    reader.read_exact(&mut buf4)?;
    let chunk_count = u32::from_le_bytes(buf4) as usize;

    let mut world = RegionChunkWorld::new();

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

fn load_world_v2<R: Read>(reader: &mut R) -> io::Result<RegionChunkWorld> {
    let base_kind = read_base_kind(reader)?;

    let mut buf4 = [0u8; 4];
    reader.read_exact(&mut buf4)?;
    let override_count = u32::from_le_bytes(buf4) as usize;

    reader.read_exact(&mut buf4)?;
    let payload_count = u32::from_le_bytes(buf4) as usize;

    let mut payloads = Vec::with_capacity(payload_count);
    for _ in 0..payload_count {
        payloads.push(read_chunk(reader)?);
    }

    let mut world = RegionChunkWorld::new_with_base(base_kind);
    for _ in 0..override_count {
        reader.read_exact(&mut buf4)?;
        let x = i32::from_le_bytes(buf4);
        reader.read_exact(&mut buf4)?;
        let y = i32::from_le_bytes(buf4);
        reader.read_exact(&mut buf4)?;
        let z = i32::from_le_bytes(buf4);
        reader.read_exact(&mut buf4)?;
        let w = i32::from_le_bytes(buf4);

        reader.read_exact(&mut buf4)?;
        let payload_index = u32::from_le_bytes(buf4) as usize;
        let Some(payload) = payloads.get(payload_index) else {
            return Err(io::Error::new(
                io::ErrorKind::InvalidData,
                "payload index out of range",
            ));
        };

        world.insert_chunk(ChunkPos::new(x, y, z, w), payload.clone());
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

            if pos != CHUNK_VOLUME {
                return Err(io::Error::new(
                    io::ErrorKind::InvalidData,
                    "RLE did not fill chunk",
                ));
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

    fn save_world_v1_legacy<W: Write>(world: &RegionChunkWorld, writer: &mut W) -> io::Result<()> {
        writer.write_all(MAGIC)?;
        writer.write_all(&VERSION_V1.to_le_bytes())?;

        let non_empty: Vec<_> = world.chunks.iter().filter(|(_, c)| !c.is_empty()).collect();
        writer.write_all(&(non_empty.len() as u32).to_le_bytes())?;

        for (&pos, chunk) in non_empty {
            writer.write_all(&pos.x.to_le_bytes())?;
            writer.write_all(&pos.y.to_le_bytes())?;
            writer.write_all(&pos.z.to_le_bytes())?;
            writer.write_all(&pos.w.to_le_bytes())?;
            write_chunk(chunk, writer)?;
        }

        Ok(())
    }

    #[test]
    fn round_trip_empty_world() {
        let world = RegionChunkWorld::new();
        let mut buf = Vec::new();
        save_world(&world, &mut buf).unwrap();
        let loaded = load_world(&mut &buf[..]).unwrap();
        assert_eq!(loaded.base_kind(), BaseWorldKind::Empty);
        assert_eq!(loaded.chunks.len(), 0);
    }

    #[test]
    fn round_trip_uniform_chunk() {
        let mut world = RegionChunkWorld::new();
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
        let mut world = RegionChunkWorld::new();
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

    #[test]
    fn round_trip_flat_floor_base_with_overrides() {
        let mut world = RegionChunkWorld::new_with_base(BaseWorldKind::FlatFloor {
            material: VoxelType(11),
        });

        // Place a floating block.
        world.set_voxel(3, 2, 1, 0, VoxelType(7));
        // Carve one floor voxel.
        world.set_voxel(0, -1, 0, 0, VoxelType::AIR);

        let mut buf = Vec::new();
        save_world(&world, &mut buf).unwrap();
        let loaded = load_world(&mut &buf[..]).unwrap();

        assert_eq!(
            loaded.base_kind(),
            BaseWorldKind::FlatFloor {
                material: VoxelType(11)
            }
        );
        assert_eq!(loaded.get_voxel(8, -1, 8, 8), VoxelType(11));
        assert_eq!(loaded.get_voxel(3, 2, 1, 0), VoxelType(7));
        assert_eq!(loaded.get_voxel(0, -1, 0, 0), VoxelType::AIR);
    }

    #[test]
    fn load_v1_legacy_world() {
        let mut world = RegionChunkWorld::new();
        let mut chunk = Chunk::new();
        chunk.set(1, 2, 3, 4, VoxelType(9));
        world.insert_chunk(ChunkPos::new(-1, 0, 2, 0), chunk);

        let mut buf = Vec::new();
        save_world_v1_legacy(&world, &mut buf).unwrap();
        let loaded = load_world(&mut &buf[..]).unwrap();

        assert_eq!(loaded.base_kind(), BaseWorldKind::Empty);
        assert_eq!(loaded.get_voxel(-7, 2, 19, 4), VoxelType(9));
    }
}
