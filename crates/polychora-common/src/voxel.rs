use std::collections::{HashMap, HashSet};
use std::io::{self, Read, Write};

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

const MAGIC: &[u8; 4] = b"V4DW";
const VERSION_V1: u32 = 1;
const VERSION_V2: u32 = 2;
const VERSION: u32 = VERSION_V2;

const ENCODING_UNIFORM: u8 = 0;
const ENCODING_RLE: u8 = 1;
const ENCODING_RAW: u8 = 2;

const BASE_KIND_EMPTY: u8 = 0;
const BASE_KIND_FLAT_FLOOR: u8 = 1;

pub fn save_world<W: Write>(world: &VoxelWorld, writer: &mut W) -> io::Result<()> {
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

pub fn load_world<R: Read>(reader: &mut R) -> io::Result<VoxelWorld> {
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

fn load_world_v1<R: Read>(reader: &mut R) -> io::Result<VoxelWorld> {
    let mut buf4 = [0u8; 4];
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

fn load_world_v2<R: Read>(reader: &mut R) -> io::Result<VoxelWorld> {
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

    let mut world = VoxelWorld::new_with_base(base_kind);
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

    fn save_world_v1_legacy<W: Write>(world: &VoxelWorld, writer: &mut W) -> io::Result<()> {
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
        let world = VoxelWorld::new();
        let mut buf = Vec::new();
        save_world(&world, &mut buf).unwrap();
        let loaded = load_world(&mut &buf[..]).unwrap();
        assert_eq!(loaded.base_kind(), BaseWorldKind::Empty);
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

    #[test]
    fn round_trip_flat_floor_base_with_overrides() {
        let mut world = VoxelWorld::new_with_base(BaseWorldKind::FlatFloor {
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
        let mut world = VoxelWorld::new();
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
