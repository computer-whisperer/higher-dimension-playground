use crate::shared::voxel::{Chunk, ChunkPos, VoxelType, CHUNK_VOLUME};
use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::fmt;
use std::sync::Arc;

pub const REGION_TILE_EDGE_CHUNKS: i32 = 4;
pub const MAX_PATCH_BYTES: usize = 1_048_576;
pub const MAX_PATCHES_PER_TICK_PER_CLIENT: usize = 16;
pub const MAX_CHUNKARRAY_CELLS_PER_NODE: usize = 256;
pub const PAGED_RLE_PAGE_EDGE_CELLS: usize = 4;
pub const QUERY_CONTENT_BOUNDS_SCAN_BUDGET_CELLS: usize = 1_048_576;

const PAGED_SPARSE_RLE_VERSION: u8 = 1;

#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub struct Aabb4i {
    pub min: [i32; 4],
    pub max: [i32; 4],
}

impl Aabb4i {
    pub fn new(min: [i32; 4], max: [i32; 4]) -> Self {
        Self { min, max }
    }

    pub fn is_valid(&self) -> bool {
        self.min[0] <= self.max[0]
            && self.min[1] <= self.max[1]
            && self.min[2] <= self.max[2]
            && self.min[3] <= self.max[3]
    }

    pub fn chunk_extents(&self) -> Option<[usize; 4]> {
        if !self.is_valid() {
            return None;
        }
        let mut extents = [0usize; 4];
        for axis in 0..4 {
            let span = i64::from(self.max[axis]) - i64::from(self.min[axis]) + 1;
            if span <= 0 {
                return None;
            }
            extents[axis] = usize::try_from(span).ok()?;
        }
        Some(extents)
    }

    pub fn chunk_cell_count(&self) -> Option<usize> {
        let extents = self.chunk_extents()?;
        extents[0]
            .checked_mul(extents[1])?
            .checked_mul(extents[2])?
            .checked_mul(extents[3])
    }

    pub fn contains_chunk(&self, pos: [i32; 4]) -> bool {
        self.is_valid()
            && pos[0] >= self.min[0]
            && pos[0] <= self.max[0]
            && pos[1] >= self.min[1]
            && pos[1] <= self.max[1]
            && pos[2] >= self.min[2]
            && pos[2] <= self.max[2]
            && pos[3] >= self.min[3]
            && pos[3] <= self.max[3]
    }

    pub fn intersects(&self, other: &Self) -> bool {
        self.is_valid()
            && other.is_valid()
            && self.min[0] <= other.max[0]
            && self.max[0] >= other.min[0]
            && self.min[1] <= other.max[1]
            && self.max[1] >= other.min[1]
            && self.min[2] <= other.max[2]
            && self.max[2] >= other.min[2]
            && self.min[3] <= other.max[3]
            && self.max[3] >= other.min[3]
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub struct ChunkKey {
    pub pos: [i32; 4],
}

impl ChunkKey {
    pub fn from_chunk_pos(chunk_pos: ChunkPos) -> Self {
        Self {
            pos: [chunk_pos.x, chunk_pos.y, chunk_pos.z, chunk_pos.w],
        }
    }

    pub fn to_chunk_pos(self) -> ChunkPos {
        ChunkPos::new(self.pos[0], self.pos[1], self.pos[2], self.pos[3])
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub struct RegionId {
    pub x: i32,
    pub y: i32,
    pub z: i32,
    pub w: i32,
}

impl RegionId {
    pub fn from_chunk_coords(coords: [i32; 4]) -> Self {
        Self {
            x: coords[0].div_euclid(REGION_TILE_EDGE_CHUNKS),
            y: coords[1].div_euclid(REGION_TILE_EDGE_CHUNKS),
            z: coords[2].div_euclid(REGION_TILE_EDGE_CHUNKS),
            w: coords[3].div_euclid(REGION_TILE_EDGE_CHUNKS),
        }
    }

    pub fn from_chunk_pos(chunk_pos: ChunkPos) -> Self {
        Self::from_chunk_coords([chunk_pos.x, chunk_pos.y, chunk_pos.z, chunk_pos.w])
    }
}

pub type RegionClockMap = HashMap<RegionId, u64>;

#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub struct NodeHandle(pub u64);

#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum QueryDetail {
    Coarse,
    Exact,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub struct QueryVolume {
    pub bounds: Aabb4i,
}

#[derive(Clone, Debug, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub struct GeneratorRef {
    pub generator_id: String,
    pub params: Vec<u8>,
    pub seed: u64,
}

#[derive(Clone, Debug, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub struct RegionTreeCore {
    pub bounds: Aabb4i,
    pub kind: RegionNodeKind,
    pub generator_version_hash: u64,
}

#[derive(Clone, Debug, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum RegionNodeKind {
    Empty,
    Uniform(u16),
    ProceduralRef(GeneratorRef),
    ChunkArray(ChunkArrayData),
    Branch(Vec<RegionTreeCore>),
}

pub type RegionWireNode = RegionTreeCore;

#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum RealizeProfile {
    Render,
    Simulation,
}

#[derive(Clone, Debug, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum ChunkPayload {
    Empty,
    Uniform(u16),
    PalettePacked {
        palette: Vec<u16>,
        bit_width: u8,
        packed_indices: Vec<u64>,
    },
    Dense16 {
        materials: Vec<u16>,
    },
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum ChunkPayloadError {
    DenseLengthMismatch { expected: usize, actual: usize },
    PaletteEmpty,
    PaletteBitWidthZeroWithMultipleEntries,
    PackedIndexOutOfRange { index: usize, palette_len: usize },
}

impl fmt::Display for ChunkPayloadError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::DenseLengthMismatch { expected, actual } => {
                write!(
                    f,
                    "dense material count mismatch: expected {expected}, got {actual}"
                )
            }
            Self::PaletteEmpty => write!(f, "palette-packed payload cannot have an empty palette"),
            Self::PaletteBitWidthZeroWithMultipleEntries => {
                write!(f, "bit_width=0 is only valid for single-entry palettes")
            }
            Self::PackedIndexOutOfRange { index, palette_len } => write!(
                f,
                "palette-packed material index {index} out of range for palette length {palette_len}"
            ),
        }
    }
}

impl std::error::Error for ChunkPayloadError {}

impl ChunkPayload {
    pub fn from_chunk_dense(chunk: &Chunk) -> Self {
        Self::Dense16 {
            materials: chunk.voxels.iter().map(|v| u16::from(v.0)).collect(),
        }
    }

    pub fn from_chunk_compact(chunk: &Chunk) -> Self {
        if chunk.is_empty() {
            return Self::Empty;
        }

        let first = u16::from(chunk.voxels[0].0);
        if chunk.voxels.iter().all(|v| u16::from(v.0) == first) {
            return Self::Uniform(first);
        }

        let mut palette = Vec::<u16>::new();
        let mut lookup = HashMap::<u16, u16>::new();
        let mut indices = Vec::<u16>::with_capacity(CHUNK_VOLUME);

        for voxel in chunk.voxels.iter() {
            let material = u16::from(voxel.0);
            let palette_idx = match lookup.get(&material) {
                Some(idx) => *idx,
                None => {
                    let next_idx = palette.len() as u16;
                    palette.push(material);
                    lookup.insert(material, next_idx);
                    next_idx
                }
            };
            indices.push(palette_idx);
        }

        let bit_width = minimal_bit_width(palette.len());
        let packed_indices = pack_indices_u16(&indices, bit_width);

        Self::PalettePacked {
            palette,
            bit_width,
            packed_indices,
        }
    }

    pub fn dense_materials(&self) -> Result<Vec<u16>, ChunkPayloadError> {
        match self {
            Self::Empty => Ok(vec![0u16; CHUNK_VOLUME]),
            Self::Uniform(material) => Ok(vec![*material; CHUNK_VOLUME]),
            Self::Dense16 { materials } => {
                if materials.len() != CHUNK_VOLUME {
                    return Err(ChunkPayloadError::DenseLengthMismatch {
                        expected: CHUNK_VOLUME,
                        actual: materials.len(),
                    });
                }
                Ok(materials.clone())
            }
            Self::PalettePacked {
                palette,
                bit_width,
                packed_indices,
            } => {
                if palette.is_empty() {
                    return Err(ChunkPayloadError::PaletteEmpty);
                }
                if *bit_width == 0 && palette.len() > 1 {
                    return Err(ChunkPayloadError::PaletteBitWidthZeroWithMultipleEntries);
                }
                let indices = unpack_indices_u16(packed_indices, *bit_width, CHUNK_VOLUME);
                let mut out = Vec::with_capacity(CHUNK_VOLUME);
                for idx in indices {
                    let idx = idx as usize;
                    if idx >= palette.len() {
                        return Err(ChunkPayloadError::PackedIndexOutOfRange {
                            index: idx,
                            palette_len: palette.len(),
                        });
                    }
                    out.push(palette[idx]);
                }
                Ok(out)
            }
        }
    }

    pub fn to_voxel_chunk(&self) -> Result<Chunk, ChunkPayloadError> {
        let dense = self.dense_materials()?;
        if dense.len() != CHUNK_VOLUME {
            return Err(ChunkPayloadError::DenseLengthMismatch {
                expected: CHUNK_VOLUME,
                actual: dense.len(),
            });
        }

        let mut chunk = Chunk::new();
        chunk.solid_count = 0;
        chunk.dirty = false;
        for (idx, material) in dense.into_iter().enumerate() {
            let voxel = VoxelType(u8::try_from(material).unwrap_or(u8::MAX));
            chunk.voxels[idx] = voxel;
            if voxel.is_solid() {
                chunk.solid_count += 1;
            }
        }
        Ok(chunk)
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum ChunkArrayIndexCodec {
    DenseU16,
    PagedSparseRle,
}

#[derive(Clone, Debug, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub struct ChunkArrayData {
    pub bounds: Aabb4i,
    pub chunk_palette: Vec<ChunkPayload>,
    pub index_codec: ChunkArrayIndexCodec,
    pub index_data: Vec<u8>,
    pub default_chunk_idx: Option<u16>,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum ChunkArrayCodecError {
    InvalidBounds,
    CellCountOverflow,
    DenseIndexLengthMismatch {
        expected_bytes: usize,
        actual_bytes: usize,
    },
    DenseIndexCountMismatch {
        expected: usize,
        actual: usize,
    },
    DefaultIndexOutOfRange {
        default_index: u16,
        palette_len: usize,
    },
    PaletteIndexOutOfRange {
        index: u16,
        palette_len: usize,
    },
    UnsupportedPagedSparseVersion {
        expected: u8,
        actual: u8,
    },
    UnsupportedPagedSparsePageEdge {
        expected: u8,
        actual: u8,
    },
    MalformedPagedSparseData,
    PageIndexOutOfRange {
        page_index: u64,
        total_pages: usize,
    },
    DuplicatePageIndex {
        page_index: u64,
    },
    ZeroRunLength,
    PageRunLengthMismatch {
        expected_cells: usize,
        decoded_cells: usize,
    },
    MissingPageWithoutDefault {
        page_index: usize,
    },
    TrailingPagedSparseBytes,
}

impl fmt::Display for ChunkArrayCodecError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::InvalidBounds => write!(f, "invalid chunk bounds"),
            Self::CellCountOverflow => write!(f, "chunk cell count overflow"),
            Self::DenseIndexLengthMismatch {
                expected_bytes,
                actual_bytes,
            } => {
                write!(
                    f,
                    "dense index byte length mismatch: expected {expected_bytes}, got {actual_bytes}"
                )
            }
            Self::DenseIndexCountMismatch { expected, actual } => {
                write!(
                    f,
                    "dense index count mismatch: expected {expected}, got {actual}"
                )
            }
            Self::DefaultIndexOutOfRange {
                default_index,
                palette_len,
            } => {
                write!(
                    f,
                    "default chunk index {default_index} out of range for palette length {palette_len}"
                )
            }
            Self::PaletteIndexOutOfRange { index, palette_len } => {
                write!(
                    f,
                    "palette index {index} out of range for palette length {palette_len}"
                )
            }
            Self::UnsupportedPagedSparseVersion { expected, actual } => {
                write!(
                    f,
                    "unsupported paged sparse version: expected {expected}, got {actual}"
                )
            }
            Self::UnsupportedPagedSparsePageEdge { expected, actual } => {
                write!(
                    f,
                    "unsupported paged sparse page edge: expected {expected}, got {actual}"
                )
            }
            Self::MalformedPagedSparseData => write!(f, "malformed paged sparse rle data"),
            Self::PageIndexOutOfRange {
                page_index,
                total_pages,
            } => {
                write!(
                    f,
                    "page index {page_index} out of range for total page count {total_pages}"
                )
            }
            Self::DuplicatePageIndex { page_index } => {
                write!(f, "duplicate page index {page_index} in paged sparse data")
            }
            Self::ZeroRunLength => write!(f, "paged sparse run length must be >= 1"),
            Self::PageRunLengthMismatch {
                expected_cells,
                decoded_cells,
            } => {
                write!(
                    f,
                    "paged sparse run length mismatch: expected {expected_cells} cells, decoded {decoded_cells}"
                )
            }
            Self::MissingPageWithoutDefault { page_index } => {
                write!(f, "missing page {page_index} without default chunk index")
            }
            Self::TrailingPagedSparseBytes => write!(f, "trailing paged sparse bytes"),
        }
    }
}

impl std::error::Error for ChunkArrayCodecError {}

impl ChunkArrayData {
    pub fn from_dense_indices(
        bounds: Aabb4i,
        chunk_palette: Vec<ChunkPayload>,
        indices: Vec<u16>,
        default_chunk_idx: Option<u16>,
    ) -> Result<Self, ChunkArrayCodecError> {
        let dims = bounds_extents(bounds)?;
        let cell_count = dims_cell_count(dims)?;
        if indices.len() != cell_count {
            return Err(ChunkArrayCodecError::DenseIndexCountMismatch {
                expected: cell_count,
                actual: indices.len(),
            });
        }

        validate_palette_indices(&indices, chunk_palette.len())?;
        validate_default_index(default_chunk_idx, chunk_palette.len())?;

        let index_data = encode_paged_sparse_rle(dims, &indices, default_chunk_idx);
        Ok(Self {
            bounds,
            chunk_palette,
            index_codec: ChunkArrayIndexCodec::PagedSparseRle,
            index_data,
            default_chunk_idx,
        })
    }

    pub fn decode_dense_indices(&self) -> Result<Vec<u16>, ChunkArrayCodecError> {
        let dims = bounds_extents(self.bounds)?;
        let cell_count = dims_cell_count(dims)?;
        validate_default_index(self.default_chunk_idx, self.chunk_palette.len())?;

        match self.index_codec {
            ChunkArrayIndexCodec::DenseU16 => {
                let expected_bytes = cell_count
                    .checked_mul(2)
                    .ok_or(ChunkArrayCodecError::CellCountOverflow)?;
                if self.index_data.len() != expected_bytes {
                    return Err(ChunkArrayCodecError::DenseIndexLengthMismatch {
                        expected_bytes,
                        actual_bytes: self.index_data.len(),
                    });
                }
                let mut out = Vec::with_capacity(cell_count);
                let mut cursor = 0usize;
                while cursor < self.index_data.len() {
                    let idx =
                        u16::from_le_bytes([self.index_data[cursor], self.index_data[cursor + 1]]);
                    out.push(idx);
                    cursor += 2;
                }
                validate_palette_indices(&out, self.chunk_palette.len())?;
                Ok(out)
            }
            ChunkArrayIndexCodec::PagedSparseRle => decode_paged_sparse_rle(
                dims,
                &self.index_data,
                self.default_chunk_idx,
                self.chunk_palette.len(),
            ),
        }
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub struct RegionClockPrecondition {
    pub region_id: RegionId,
    pub expected_clock: u64,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub struct RegionClockUpdate {
    pub region_id: RegionId,
    pub new_clock: u64,
}

#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct RegionSubtreePatch {
    pub patch_seq: u64,
    pub bounds: Aabb4i,
    pub preconditions: Vec<RegionClockPrecondition>,
    pub clock_updates: Vec<RegionClockUpdate>,
    pub subtree: RegionWireNode,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub struct RegionResyncEntry {
    pub region_id: RegionId,
    pub client_clock: u64,
}

#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct RegionResyncRequest {
    pub regions: Vec<RegionResyncEntry>,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub struct WorldFieldCapabilities {
    pub protocol_version: u16,
    pub generator_manifest_hash: u64,
    pub feature_bits: u64,
}

pub trait WorldField {
    fn query_region_core(&self, query: QueryVolume, detail: QueryDetail) -> Arc<RegionTreeCore>;

    fn query_content_bounds(&self, query: QueryVolume) -> Option<Aabb4i>;

    fn realize_chunk(&mut self, key: ChunkKey, profile: RealizeProfile) -> ChunkPayload;
}

mod region_tree;
pub use region_tree::{
    collect_non_empty_chunks_from_core_in_bounds, slice_non_empty_region_core_in_bounds,
    slice_region_core_in_bounds, RegionChunkTree,
};

fn minimal_bit_width(palette_len: usize) -> u8 {
    if palette_len <= 1 {
        return 0;
    }
    let mut bits = 0u8;
    let mut max_value = palette_len - 1;
    while max_value > 0 {
        bits += 1;
        max_value >>= 1;
    }
    bits
}

fn pack_indices_u16(indices: &[u16], bit_width: u8) -> Vec<u64> {
    if bit_width == 0 {
        return Vec::new();
    }
    let total_bits = indices.len() * bit_width as usize;
    let word_count = (total_bits + 63) / 64;
    let mut out = vec![0u64; word_count];

    let mask = if bit_width == 64 {
        u64::MAX
    } else {
        (1u64 << bit_width) - 1
    };

    let mut bit_cursor = 0usize;
    for index in indices {
        let value = u64::from(*index) & mask;
        let word_idx = bit_cursor / 64;
        let bit_off = bit_cursor % 64;
        out[word_idx] |= value << bit_off;
        if bit_off + bit_width as usize > 64 {
            let spill_bits = bit_off + bit_width as usize - 64;
            out[word_idx + 1] |= value >> (bit_width as usize - spill_bits);
        }
        bit_cursor += bit_width as usize;
    }

    out
}

fn unpack_indices_u16(packed: &[u64], bit_width: u8, count: usize) -> Vec<u16> {
    if bit_width == 0 {
        return vec![0u16; count];
    }
    let mask = if bit_width == 64 {
        u64::MAX
    } else {
        (1u64 << bit_width) - 1
    };

    let mut out = Vec::with_capacity(count);
    let mut bit_cursor = 0usize;
    for _ in 0..count {
        let word_idx = bit_cursor / 64;
        let bit_off = bit_cursor % 64;

        let mut value = (packed[word_idx] >> bit_off) & mask;
        if bit_off + bit_width as usize > 64 {
            let spill_bits = bit_off + bit_width as usize - 64;
            let high = packed[word_idx + 1] & ((1u64 << spill_bits) - 1);
            value |= high << (bit_width as usize - spill_bits);
        }

        out.push(value as u16);
        bit_cursor += bit_width as usize;
    }
    out
}

fn bounds_extents(bounds: Aabb4i) -> Result<[usize; 4], ChunkArrayCodecError> {
    bounds
        .chunk_extents()
        .ok_or(ChunkArrayCodecError::InvalidBounds)
}

fn dims_cell_count(dims: [usize; 4]) -> Result<usize, ChunkArrayCodecError> {
    dims[0]
        .checked_mul(dims[1])
        .and_then(|v| v.checked_mul(dims[2]))
        .and_then(|v| v.checked_mul(dims[3]))
        .ok_or(ChunkArrayCodecError::CellCountOverflow)
}

fn validate_default_index(
    default_idx: Option<u16>,
    palette_len: usize,
) -> Result<(), ChunkArrayCodecError> {
    if let Some(default_idx) = default_idx {
        if usize::from(default_idx) >= palette_len {
            return Err(ChunkArrayCodecError::DefaultIndexOutOfRange {
                default_index: default_idx,
                palette_len,
            });
        }
    }
    Ok(())
}

fn validate_palette_indices(
    indices: &[u16],
    palette_len: usize,
) -> Result<(), ChunkArrayCodecError> {
    for idx in indices {
        if usize::from(*idx) >= palette_len {
            return Err(ChunkArrayCodecError::PaletteIndexOutOfRange {
                index: *idx,
                palette_len,
            });
        }
    }
    Ok(())
}

fn encode_paged_sparse_rle(dims: [usize; 4], indices: &[u16], default_idx: Option<u16>) -> Vec<u8> {
    let edge = PAGED_RLE_PAGE_EDGE_CELLS;
    let pages_dims = [
        ceil_div(dims[0], edge),
        ceil_div(dims[1], edge),
        ceil_div(dims[2], edge),
        ceil_div(dims[3], edge),
    ];

    let mut encoded_pages = Vec::<(u64, Vec<(u16, u16)>)>::new();

    for pw in 0..pages_dims[3] {
        for pz in 0..pages_dims[2] {
            for py in 0..pages_dims[1] {
                for px in 0..pages_dims[0] {
                    let page_linear_index = linear_page_index([px, py, pz, pw], pages_dims);
                    let page_cells =
                        gather_page_indices(dims, pages_dims, [px, py, pz, pw], indices);
                    if let Some(default_idx) = default_idx {
                        if page_cells.iter().all(|idx| *idx == default_idx) {
                            continue;
                        }
                    }

                    let runs = rle_runs(&page_cells);
                    encoded_pages.push((page_linear_index as u64, runs));
                }
            }
        }
    }

    let mut out = Vec::new();
    out.push(PAGED_SPARSE_RLE_VERSION);
    out.push(PAGED_RLE_PAGE_EDGE_CELLS as u8);
    out.extend_from_slice(&0u16.to_le_bytes());
    out.extend_from_slice(&(dims[0] as u32).to_le_bytes());
    out.extend_from_slice(&(dims[1] as u32).to_le_bytes());
    out.extend_from_slice(&(dims[2] as u32).to_le_bytes());
    out.extend_from_slice(&(dims[3] as u32).to_le_bytes());
    out.extend_from_slice(&(encoded_pages.len() as u32).to_le_bytes());

    for (page_idx, runs) in encoded_pages {
        out.extend_from_slice(&page_idx.to_le_bytes());
        out.extend_from_slice(&(runs.len() as u16).to_le_bytes());
        for (run_len, palette_idx) in runs {
            out.extend_from_slice(&run_len.to_le_bytes());
            out.extend_from_slice(&palette_idx.to_le_bytes());
        }
    }

    out
}

fn decode_paged_sparse_rle(
    dims: [usize; 4],
    data: &[u8],
    default_idx: Option<u16>,
    palette_len: usize,
) -> Result<Vec<u16>, ChunkArrayCodecError> {
    const HEADER_LEN: usize = 24;

    if data.len() < HEADER_LEN {
        return Err(ChunkArrayCodecError::MalformedPagedSparseData);
    }

    let version = data[0];
    if version != PAGED_SPARSE_RLE_VERSION {
        return Err(ChunkArrayCodecError::UnsupportedPagedSparseVersion {
            expected: PAGED_SPARSE_RLE_VERSION,
            actual: version,
        });
    }

    let page_edge = data[1];
    if page_edge != PAGED_RLE_PAGE_EDGE_CELLS as u8 {
        return Err(ChunkArrayCodecError::UnsupportedPagedSparsePageEdge {
            expected: PAGED_RLE_PAGE_EDGE_CELLS as u8,
            actual: page_edge,
        });
    }

    let header_dims = [
        u32::from_le_bytes([data[4], data[5], data[6], data[7]]) as usize,
        u32::from_le_bytes([data[8], data[9], data[10], data[11]]) as usize,
        u32::from_le_bytes([data[12], data[13], data[14], data[15]]) as usize,
        u32::from_le_bytes([data[16], data[17], data[18], data[19]]) as usize,
    ];
    if header_dims != dims {
        return Err(ChunkArrayCodecError::MalformedPagedSparseData);
    }

    let emitted_page_count = u32::from_le_bytes([data[20], data[21], data[22], data[23]]) as usize;

    let edge = PAGED_RLE_PAGE_EDGE_CELLS;
    let pages_dims = [
        ceil_div(dims[0], edge),
        ceil_div(dims[1], edge),
        ceil_div(dims[2], edge),
        ceil_div(dims[3], edge),
    ];
    let total_pages = dims_cell_count(pages_dims)?;

    let cell_count = dims_cell_count(dims)?;
    let default_fill = default_idx.unwrap_or(0);
    let mut dense = vec![default_fill; cell_count];
    let mut seen_pages = vec![false; total_pages];

    let mut cursor = HEADER_LEN;
    for _ in 0..emitted_page_count {
        if cursor + 10 > data.len() {
            return Err(ChunkArrayCodecError::MalformedPagedSparseData);
        }

        let page_idx = u64::from_le_bytes([
            data[cursor],
            data[cursor + 1],
            data[cursor + 2],
            data[cursor + 3],
            data[cursor + 4],
            data[cursor + 5],
            data[cursor + 6],
            data[cursor + 7],
        ]);
        cursor += 8;

        let run_count = u16::from_le_bytes([data[cursor], data[cursor + 1]]) as usize;
        cursor += 2;

        let page_idx_usize =
            usize::try_from(page_idx).map_err(|_| ChunkArrayCodecError::PageIndexOutOfRange {
                page_index: page_idx,
                total_pages,
            })?;
        if page_idx_usize >= total_pages {
            return Err(ChunkArrayCodecError::PageIndexOutOfRange {
                page_index: page_idx,
                total_pages,
            });
        }
        if seen_pages[page_idx_usize] {
            return Err(ChunkArrayCodecError::DuplicatePageIndex {
                page_index: page_idx,
            });
        }
        seen_pages[page_idx_usize] = true;

        let page_cell_linear = page_linear_cells(dims, pages_dims, page_idx_usize);
        if page_cell_linear.is_empty() {
            return Err(ChunkArrayCodecError::MalformedPagedSparseData);
        }

        let mut decoded_cells = 0usize;
        for _ in 0..run_count {
            if cursor + 4 > data.len() {
                return Err(ChunkArrayCodecError::MalformedPagedSparseData);
            }
            let run_len = u16::from_le_bytes([data[cursor], data[cursor + 1]]) as usize;
            let palette_idx = u16::from_le_bytes([data[cursor + 2], data[cursor + 3]]);
            cursor += 4;

            if run_len == 0 {
                return Err(ChunkArrayCodecError::ZeroRunLength);
            }
            if usize::from(palette_idx) >= palette_len {
                return Err(ChunkArrayCodecError::PaletteIndexOutOfRange {
                    index: palette_idx,
                    palette_len,
                });
            }

            let end = decoded_cells.saturating_add(run_len);
            if end > page_cell_linear.len() {
                return Err(ChunkArrayCodecError::PageRunLengthMismatch {
                    expected_cells: page_cell_linear.len(),
                    decoded_cells: end,
                });
            }
            for cell_idx in &page_cell_linear[decoded_cells..end] {
                dense[*cell_idx] = palette_idx;
            }
            decoded_cells = end;
        }

        if decoded_cells != page_cell_linear.len() {
            return Err(ChunkArrayCodecError::PageRunLengthMismatch {
                expected_cells: page_cell_linear.len(),
                decoded_cells,
            });
        }
    }

    if cursor != data.len() {
        return Err(ChunkArrayCodecError::TrailingPagedSparseBytes);
    }

    if default_idx.is_none() {
        for (page_idx, seen) in seen_pages.into_iter().enumerate() {
            if !seen {
                return Err(ChunkArrayCodecError::MissingPageWithoutDefault {
                    page_index: page_idx,
                });
            }
        }
    }

    Ok(dense)
}

fn gather_page_indices(
    dims: [usize; 4],
    pages_dims: [usize; 4],
    page_coords: [usize; 4],
    indices: &[u16],
) -> Vec<u16> {
    let mut out = Vec::new();
    let edge = PAGED_RLE_PAGE_EDGE_CELLS;

    debug_assert!(page_coords[0] < pages_dims[0]);
    debug_assert!(page_coords[1] < pages_dims[1]);
    debug_assert!(page_coords[2] < pages_dims[2]);
    debug_assert!(page_coords[3] < pages_dims[3]);

    for lw in 0..edge {
        let gw = page_coords[3] * edge + lw;
        if gw >= dims[3] {
            break;
        }
        for lz in 0..edge {
            let gz = page_coords[2] * edge + lz;
            if gz >= dims[2] {
                break;
            }
            for ly in 0..edge {
                let gy = page_coords[1] * edge + ly;
                if gy >= dims[1] {
                    break;
                }
                for lx in 0..edge {
                    let gx = page_coords[0] * edge + lx;
                    if gx >= dims[0] {
                        break;
                    }
                    let linear = linear_cell_index([gx, gy, gz, gw], dims);
                    out.push(indices[linear]);
                }
            }
        }
    }

    out
}

fn page_linear_cells(dims: [usize; 4], pages_dims: [usize; 4], page_idx: usize) -> Vec<usize> {
    let coords = page_coords_from_linear(page_idx, pages_dims);
    let edge = PAGED_RLE_PAGE_EDGE_CELLS;
    let mut out = Vec::new();

    for lw in 0..edge {
        let gw = coords[3] * edge + lw;
        if gw >= dims[3] {
            break;
        }
        for lz in 0..edge {
            let gz = coords[2] * edge + lz;
            if gz >= dims[2] {
                break;
            }
            for ly in 0..edge {
                let gy = coords[1] * edge + ly;
                if gy >= dims[1] {
                    break;
                }
                for lx in 0..edge {
                    let gx = coords[0] * edge + lx;
                    if gx >= dims[0] {
                        break;
                    }
                    out.push(linear_cell_index([gx, gy, gz, gw], dims));
                }
            }
        }
    }

    out
}

fn rle_runs(values: &[u16]) -> Vec<(u16, u16)> {
    if values.is_empty() {
        return Vec::new();
    }
    let mut runs = Vec::<(u16, u16)>::new();
    let mut current = values[0];
    let mut len = 1u16;

    for value in values.iter().skip(1) {
        if *value == current && len < u16::MAX {
            len += 1;
            continue;
        }
        runs.push((len, current));
        current = *value;
        len = 1;
    }
    runs.push((len, current));
    runs
}

fn ceil_div(value: usize, divisor: usize) -> usize {
    (value + divisor - 1) / divisor
}

fn linear_cell_index(coords: [usize; 4], dims: [usize; 4]) -> usize {
    coords[0] + dims[0] * (coords[1] + dims[1] * (coords[2] + dims[2] * coords[3]))
}

fn linear_page_index(coords: [usize; 4], pages_dims: [usize; 4]) -> usize {
    coords[0]
        + pages_dims[0] * (coords[1] + pages_dims[1] * (coords[2] + pages_dims[2] * coords[3]))
}

fn page_coords_from_linear(mut index: usize, pages_dims: [usize; 4]) -> [usize; 4] {
    let x = index % pages_dims[0];
    index /= pages_dims[0];
    let y = index % pages_dims[1];
    index /= pages_dims[1];
    let z = index % pages_dims[2];
    index /= pages_dims[2];
    let w = index;
    [x, y, z, w]
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::shared::voxel::{Chunk, VoxelType, CHUNK_SIZE};
    use crate::shared::worldfield_testkit::{
        assert_tree_matches_reference, random_chunk_key_in_bounds, random_chunk_payload,
        random_sub_bounds, DeterministicRng, ReferenceChunkStore,
    };

    fn key(x: i32, y: i32, z: i32, w: i32) -> ChunkKey {
        ChunkKey { pos: [x, y, z, w] }
    }

    #[test]
    fn region_id_mapping_uses_floor_division_for_negatives() {
        let id = RegionId::from_chunk_coords([-1, -4, -5, 7]);
        assert_eq!(id.x, -1);
        assert_eq!(id.y, -1);
        assert_eq!(id.z, -2);
        assert_eq!(id.w, 1);
    }

    #[test]
    fn aabb_chunk_cell_count_matches_extents() {
        let bounds = Aabb4i::new([2, -1, 0, 3], [4, 1, 1, 4]);
        assert_eq!(bounds.chunk_extents(), Some([3, 3, 2, 2]));
        assert_eq!(bounds.chunk_cell_count(), Some(36));
    }

    #[test]
    fn chunk_payload_compact_roundtrip_matches_dense_materials() {
        let mut chunk = Chunk::new();
        chunk.set(0, 0, 0, 0, VoxelType(7));
        chunk.set(1, 0, 0, 0, VoxelType(9));
        chunk.set(2, 0, 0, 0, VoxelType(7));

        let payload = ChunkPayload::from_chunk_compact(&chunk);
        let dense = payload
            .dense_materials()
            .expect("compact payload should decode to dense materials");

        assert_eq!(dense.len(), CHUNK_VOLUME);
        assert_eq!(dense[Chunk::local_index(0, 0, 0, 0)], 7);
        assert_eq!(dense[Chunk::local_index(1, 0, 0, 0)], 9);
        assert_eq!(dense[Chunk::local_index(2, 0, 0, 0)], 7);
    }

    #[test]
    fn chunk_payload_to_voxel_chunk_roundtrip_preserves_values() {
        let mut chunk = Chunk::new();
        chunk.set(0, 0, 0, 0, VoxelType(3));
        chunk.set(1, 0, 0, 0, VoxelType(11));
        chunk.set(2, CHUNK_SIZE - 1, 0, 0, VoxelType(27));

        let payload = ChunkPayload::from_chunk_compact(&chunk);
        let roundtrip = payload.to_voxel_chunk().expect("decode to voxel chunk");

        assert_eq!(
            roundtrip.voxels[Chunk::local_index(0, 0, 0, 0)],
            VoxelType(3)
        );
        assert_eq!(
            roundtrip.voxels[Chunk::local_index(1, 0, 0, 0)],
            VoxelType(11)
        );
        assert_eq!(
            roundtrip.voxels[Chunk::local_index(2, CHUNK_SIZE - 1, 0, 0)],
            VoxelType(27)
        );
    }

    #[test]
    fn region_chunk_tree_set_get_and_remove_single_chunk() {
        let mut tree = RegionChunkTree::new();
        assert!(!tree.has_chunk(key(0, 0, 0, 0)));

        assert!(tree.set_chunk(key(0, 0, 0, 0), Some(ChunkPayload::Uniform(12))));
        assert!(tree.has_chunk(key(0, 0, 0, 0)));
        assert_eq!(
            tree.chunk_payload(key(0, 0, 0, 0)),
            Some(ChunkPayload::Uniform(12))
        );

        assert!(tree.remove_chunk(key(0, 0, 0, 0)));
        assert!(!tree.has_chunk(key(0, 0, 0, 0)));
        assert!(tree.root().is_none());
    }

    #[test]
    fn region_chunk_tree_merges_uniform_and_fragments_on_edit() {
        let mut tree = RegionChunkTree::new();
        assert!(tree.set_chunk(key(0, 0, 0, 0), Some(ChunkPayload::Uniform(7))));
        assert!(tree.set_chunk(key(1, 0, 0, 0), Some(ChunkPayload::Uniform(7))));

        let root = tree.root().expect("root exists");
        assert!(matches!(root.kind, RegionNodeKind::Uniform(7)));

        assert!(tree.set_chunk(key(0, 0, 0, 0), Some(ChunkPayload::Uniform(9))));
        assert_eq!(
            tree.chunk_payload(key(0, 0, 0, 0)),
            Some(ChunkPayload::Uniform(9))
        );
        assert_eq!(
            tree.chunk_payload(key(1, 0, 0, 0)),
            Some(ChunkPayload::Uniform(7))
        );

        assert!(tree.set_chunk(key(0, 0, 0, 0), Some(ChunkPayload::Uniform(7))));
        let root = tree.root().expect("root exists");
        assert!(matches!(root.kind, RegionNodeKind::Uniform(7)));
    }

    #[test]
    fn region_chunk_tree_expands_for_distant_insertions() {
        let mut tree = RegionChunkTree::new();
        assert!(tree.set_chunk(key(0, 0, 0, 0), Some(ChunkPayload::Uniform(1))));
        assert!(tree.set_chunk(key(48, -7, 13, 21), Some(ChunkPayload::Uniform(2))));

        assert_eq!(
            tree.chunk_payload(key(0, 0, 0, 0)),
            Some(ChunkPayload::Uniform(1))
        );
        assert_eq!(
            tree.chunk_payload(key(48, -7, 13, 21)),
            Some(ChunkPayload::Uniform(2))
        );
        let root = tree.root().expect("root exists");
        assert!(root.bounds.contains_chunk([0, 0, 0, 0]));
        assert!(root.bounds.contains_chunk([48, -7, 13, 21]));
    }

    #[test]
    fn region_chunk_tree_preserves_explicit_empty_chunk() {
        let mut tree = RegionChunkTree::new();
        assert!(tree.set_chunk(key(2, -1, 0, 0), Some(ChunkPayload::Empty)));
        assert!(tree.has_chunk(key(2, -1, 0, 0)));

        let payload = tree
            .chunk_payload(key(2, -1, 0, 0))
            .expect("empty chunk payload exists");
        let dense = payload.dense_materials().expect("dense decode");
        assert!(dense.iter().all(|v| *v == 0));
    }

    #[test]
    fn region_chunk_tree_non_empty_bounds_ignores_air_chunks() {
        let mut tree = RegionChunkTree::new();
        assert!(tree.set_chunk(key(0, 0, 0, 0), Some(ChunkPayload::Empty)));
        let around_origin = Aabb4i::new([-1, -1, -1, -1], [1, 1, 1, 1]);
        assert!(!tree.any_non_empty_chunk_in_bounds(around_origin));

        assert!(tree.set_chunk(key(4, 0, 0, 0), Some(ChunkPayload::Uniform(6))));
        let around_solid = Aabb4i::new([3, -1, -1, -1], [5, 1, 1, 1]);
        assert!(tree.any_non_empty_chunk_in_bounds(around_solid));
    }

    #[test]
    fn region_chunk_tree_roundtrips_non_uniform_payload() {
        let mut chunk = Chunk::new();
        chunk.set(0, 0, 0, 0, VoxelType(4));
        chunk.set(1, 0, 0, 0, VoxelType(9));
        chunk.set(2, 0, 0, 0, VoxelType(4));
        let payload = ChunkPayload::from_chunk_compact(&chunk);
        let expected = payload.dense_materials().expect("dense expected");

        let mut tree = RegionChunkTree::new();
        assert!(tree.set_chunk(key(7, 1, -2, 3), Some(payload)));
        let roundtrip = tree
            .chunk_payload(key(7, 1, -2, 3))
            .expect("payload exists");
        let dense = roundtrip.dense_materials().expect("dense roundtrip");
        assert_eq!(dense, expected);
    }

    #[test]
    fn region_chunk_tree_collects_chunks_and_counts_non_empty() {
        let mut tree = RegionChunkTree::new();
        assert!(tree.set_chunk(key(0, 0, 0, 0), Some(ChunkPayload::Uniform(0))));
        assert!(tree.set_chunk(key(1, 0, 0, 0), Some(ChunkPayload::Uniform(3))));
        assert!(tree.set_chunk(key(2, 0, 0, 0), Some(ChunkPayload::Uniform(5))));

        let mut collected = tree.collect_chunks();
        collected.sort_by_key(|(key, _)| key.pos);
        assert_eq!(collected.len(), 3);
        assert_eq!(collected[0].0.pos, [0, 0, 0, 0]);
        assert_eq!(collected[1].0.pos, [1, 0, 0, 0]);
        assert_eq!(collected[2].0.pos, [2, 0, 0, 0]);
        assert_eq!(tree.non_empty_chunk_count(), 2);
    }

    #[test]
    fn region_chunk_tree_collect_chunks_in_bounds_filters_results() {
        let mut tree = RegionChunkTree::new();
        assert!(tree.set_chunk(key(-3, 0, 0, 0), Some(ChunkPayload::Uniform(2))));
        assert!(tree.set_chunk(key(0, 0, 0, 0), Some(ChunkPayload::Uniform(4))));
        assert!(tree.set_chunk(key(5, 0, 0, 0), Some(ChunkPayload::Uniform(6))));

        let bounds = Aabb4i::new([-1, -1, -1, -1], [2, 1, 1, 1]);
        let mut collected = tree.collect_chunks_in_bounds(bounds);
        collected.sort_by_key(|(key, _)| key.pos);

        assert_eq!(collected.len(), 1);
        assert_eq!(collected[0].0.pos, [0, 0, 0, 0]);
        assert_eq!(collected[0].1, ChunkPayload::Uniform(4));
    }

    #[test]
    fn collect_non_empty_chunks_from_core_in_bounds_skips_air_and_procedural() {
        let bounds = Aabb4i::new([0, 0, 0, 0], [2, 0, 0, 0]);
        let core = RegionTreeCore {
            bounds,
            kind: RegionNodeKind::Branch(vec![
                RegionTreeCore {
                    bounds: Aabb4i::new([0, 0, 0, 0], [0, 0, 0, 0]),
                    kind: RegionNodeKind::Uniform(5),
                    generator_version_hash: 0,
                },
                RegionTreeCore {
                    bounds: Aabb4i::new([1, 0, 0, 0], [1, 0, 0, 0]),
                    kind: RegionNodeKind::Uniform(0),
                    generator_version_hash: 0,
                },
                RegionTreeCore {
                    bounds: Aabb4i::new([2, 0, 0, 0], [2, 0, 0, 0]),
                    kind: RegionNodeKind::ProceduralRef(GeneratorRef {
                        generator_id: "test.gen".to_string(),
                        params: vec![1, 2, 3],
                        seed: 42,
                    }),
                    generator_version_hash: 0,
                },
            ]),
            generator_version_hash: 0,
        };

        let collected = collect_non_empty_chunks_from_core_in_bounds(&core, bounds);
        assert_eq!(collected, vec![(key(0, 0, 0, 0), ChunkPayload::Uniform(5))]);
    }

    #[test]
    fn collect_non_empty_chunks_from_core_in_bounds_filters_chunk_array_cells() {
        let bounds = Aabb4i::new([0, 0, 0, 0], [1, 0, 0, 0]);
        let chunk_array = ChunkArrayData::from_dense_indices(
            bounds,
            vec![ChunkPayload::Empty, ChunkPayload::Uniform(9)],
            vec![1, 0],
            Some(0),
        )
        .expect("chunk array encoding");
        let core = RegionTreeCore {
            bounds,
            kind: RegionNodeKind::ChunkArray(chunk_array),
            generator_version_hash: 0,
        };

        let left_only = collect_non_empty_chunks_from_core_in_bounds(
            &core,
            Aabb4i::new([0, 0, 0, 0], [0, 0, 0, 0]),
        );
        assert_eq!(left_only, vec![(key(0, 0, 0, 0), ChunkPayload::Uniform(9))]);

        let right_only = collect_non_empty_chunks_from_core_in_bounds(
            &core,
            Aabb4i::new([1, 0, 0, 0], [1, 0, 0, 0]),
        );
        assert!(right_only.is_empty());
    }

    #[test]
    fn region_chunk_tree_splice_non_empty_core_in_bounds_replaces_window_contents() {
        let mut tree = RegionChunkTree::new();
        assert!(tree.set_chunk(key(0, 0, 0, 0), Some(ChunkPayload::Uniform(2))));
        assert!(tree.set_chunk(key(2, 0, 0, 0), Some(ChunkPayload::Uniform(4))));

        let bounds = Aabb4i::new([0, 0, 0, 0], [2, 0, 0, 0]);
        let chunk_array = ChunkArrayData::from_dense_indices(
            bounds,
            vec![ChunkPayload::Empty, ChunkPayload::Uniform(9)],
            vec![0, 1, 0],
            Some(0),
        )
        .expect("chunk array encoding");
        let core = RegionTreeCore {
            bounds,
            kind: RegionNodeKind::ChunkArray(chunk_array),
            generator_version_hash: 0,
        };

        let changed_bounds = tree.splice_non_empty_core_in_bounds(bounds, &core);
        assert_eq!(changed_bounds, Some(bounds));

        let mut non_empty = collect_non_empty_chunks_from_core_in_bounds(
            &tree.slice_non_empty_core_in_bounds(bounds),
            bounds,
        );
        non_empty.sort_unstable_by_key(|(chunk_key, _)| chunk_key.pos);
        assert_eq!(non_empty, vec![(key(1, 0, 0, 0), ChunkPayload::Uniform(9))]);
    }

    #[test]
    fn region_chunk_tree_slice_core_in_bounds_preserves_query_bounds() {
        let mut tree = RegionChunkTree::new();
        assert!(tree.set_chunk(key(0, 0, 0, 0), Some(ChunkPayload::Uniform(2))));
        assert!(tree.set_chunk(key(2, 0, 0, 0), Some(ChunkPayload::Uniform(4))));

        let bounds = Aabb4i::new([1, 0, 0, 0], [3, 0, 0, 0]);
        let slice = tree.slice_core_in_bounds(bounds);
        assert_eq!(slice.bounds, bounds);

        let mut collected = collect_non_empty_chunks_from_core_in_bounds(&slice, bounds);
        collected.sort_unstable_by_key(|(key, _)| key.pos);
        assert_eq!(collected, vec![(key(2, 0, 0, 0), ChunkPayload::Uniform(4))]);
    }

    #[test]
    fn region_chunk_tree_slice_non_empty_core_in_bounds_prunes_empty_coverage() {
        let mut tree = RegionChunkTree::new();
        assert!(tree.set_chunk(key(1, 0, 0, 0), Some(ChunkPayload::Uniform(9))));

        let bounds = Aabb4i::new([0, 0, 0, 0], [2, 0, 0, 0]);
        let slice = tree.slice_non_empty_core_in_bounds(bounds);
        let mut collected = collect_non_empty_chunks_from_core_in_bounds(&slice, bounds);
        collected.sort_unstable_by_key(|(key, _)| key.pos);
        assert_eq!(collected, vec![(key(1, 0, 0, 0), ChunkPayload::Uniform(9))]);
    }

    #[test]
    fn region_chunk_tree_take_non_empty_core_in_bounds_extracts_and_clears() {
        let mut tree = RegionChunkTree::new();
        assert!(tree.set_chunk(key(0, 0, 0, 0), Some(ChunkPayload::Uniform(2))));
        assert!(tree.set_chunk(key(1, 0, 0, 0), Some(ChunkPayload::Uniform(3))));
        assert!(tree.set_chunk(key(2, 0, 0, 0), Some(ChunkPayload::Uniform(4))));

        let bounds = Aabb4i::new([1, 0, 0, 0], [2, 0, 0, 0]);
        let extracted = tree.take_non_empty_core_in_bounds(bounds);
        let mut extracted_chunks = collect_non_empty_chunks_from_core_in_bounds(&extracted, bounds);
        extracted_chunks.sort_unstable_by_key(|(chunk_key, _)| chunk_key.pos);
        assert_eq!(
            extracted_chunks,
            vec![
                (key(1, 0, 0, 0), ChunkPayload::Uniform(3)),
                (key(2, 0, 0, 0), ChunkPayload::Uniform(4))
            ]
        );

        assert_eq!(
            tree.chunk_payload(key(0, 0, 0, 0)),
            Some(ChunkPayload::Uniform(2))
        );
        assert!(!tree.has_chunk(key(1, 0, 0, 0)));
        assert!(!tree.has_chunk(key(2, 0, 0, 0)));
    }

    #[test]
    fn region_chunk_tree_lazy_drop_outside_bounds_respects_budget() {
        let mut tree = RegionChunkTree::new();
        assert!(tree.set_chunk(key(-8, 0, 0, 0), Some(ChunkPayload::Uniform(6))));
        assert!(tree.set_chunk(key(8, 0, 0, 0), Some(ChunkPayload::Uniform(7))));
        assert_eq!(tree.non_empty_chunk_count(), 2);

        let keep = Aabb4i::new([0, 0, 0, 0], [0, 0, 0, 0]);
        assert!(tree.lazy_drop_outside_bounds(keep, 0).is_none());
        assert_eq!(tree.non_empty_chunk_count(), 2);

        let first_drop = tree.lazy_drop_outside_bounds(keep, 1);
        assert!(first_drop.is_some());
        assert_eq!(tree.non_empty_chunk_count(), 1);

        for _ in 0..32 {
            if tree.non_empty_chunk_count() == 0 {
                break;
            }
            let _ = tree.lazy_drop_outside_bounds(keep, 4);
        }
        assert_eq!(tree.non_empty_chunk_count(), 0);
    }

    #[test]
    fn region_chunk_tree_randomized_splice_matches_reference_model() {
        let domain = Aabb4i::new([-10, -4, -10, -10], [10, 4, 10, 10]);
        for seed in [0x534c_4943_455f_31_u64, 0x534c_4943_455f_32] {
            let mut rng = DeterministicRng::new(seed);
            let mut baseline = RegionChunkTree::new();

            for _ in 0..200 {
                let key = random_chunk_key_in_bounds(&mut rng, domain);
                let payload = if rng.chance(1, 5) {
                    None
                } else {
                    Some(random_chunk_payload(&mut rng))
                };
                let _ = baseline.set_chunk(key, payload);
            }

            let mut splice_tree = baseline.clone();
            let mut reference = ReferenceChunkStore::from_tree(&baseline);

            for step in 0..120 {
                let bounds = random_sub_bounds(&mut rng, domain);
                let desired_count = rng.range_usize(0, 24);
                let mut desired = Vec::with_capacity(desired_count);
                for _ in 0..desired_count {
                    desired.push((
                        random_chunk_key_in_bounds(&mut rng, bounds),
                        random_chunk_payload(&mut rng),
                    ));
                }

                let desired_tree = RegionChunkTree::from_chunks(desired);
                let core = desired_tree.slice_core_in_bounds(bounds);
                let desired_non_empty = collect_non_empty_chunks_from_core_in_bounds(&core, bounds);
                let mut before_in_bounds = splice_tree.collect_chunks_in_bounds(bounds);
                before_in_bounds.sort_unstable_by_key(|(chunk_key, _)| chunk_key.pos);
                let mut desired_in_bounds = desired_non_empty.clone();
                desired_in_bounds.sort_unstable_by_key(|(chunk_key, _)| chunk_key.pos);
                let expected_changed = before_in_bounds != desired_in_bounds;

                let changed_bounds = splice_tree.splice_non_empty_core_in_bounds(bounds, &core);
                reference.replace_non_empty_chunks_in_bounds(bounds, desired_non_empty);

                let mut expected = reference.collect_chunks_sorted();
                expected.sort_unstable_by_key(|(chunk_key, _)| chunk_key.pos);
                let mut actual = splice_tree.collect_chunks();
                actual.sort_unstable_by_key(|(chunk_key, _)| chunk_key.pos);
                assert_eq!(actual, expected, "splice mismatch seed={seed} step={step}");
                assert_eq!(
                    changed_bounds.is_some(),
                    expected_changed,
                    "splice changed_bounds mismatch seed={seed} step={step}"
                );
            }
        }
    }

    #[test]
    fn chunk_array_paged_sparse_roundtrip_preserves_indices() {
        let bounds = Aabb4i::new([0, 0, 0, 0], [5, 3, 2, 1]);
        let cell_count = bounds.chunk_cell_count().expect("valid bounds");

        let mut indices = vec![0u16; cell_count];
        indices[0] = 1;
        indices[9] = 2;
        indices[cell_count - 1] = 3;

        let chunk_palette = vec![
            ChunkPayload::Empty,
            ChunkPayload::Uniform(10),
            ChunkPayload::Uniform(20),
            ChunkPayload::Uniform(30),
        ];

        let chunk_array =
            ChunkArrayData::from_dense_indices(bounds, chunk_palette, indices.clone(), Some(0))
                .expect("encoding chunk array");

        assert_eq!(
            chunk_array.index_codec,
            ChunkArrayIndexCodec::PagedSparseRle
        );

        let decoded = chunk_array
            .decode_dense_indices()
            .expect("decoding chunk array");
        assert_eq!(decoded, indices);
    }

    #[test]
    fn chunk_array_missing_pages_without_default_is_rejected() {
        let bounds = Aabb4i::new([0, 0, 0, 0], [7, 3, 3, 3]);
        let cell_count = bounds.chunk_cell_count().expect("valid bounds");
        let mut indices = vec![0u16; cell_count];
        indices[0] = 1;

        let mut chunk_array = ChunkArrayData::from_dense_indices(
            bounds,
            vec![ChunkPayload::Empty, ChunkPayload::Uniform(1)],
            indices,
            Some(0),
        )
        .expect("encoding chunk array");

        chunk_array.default_chunk_idx = None;

        let error = chunk_array
            .decode_dense_indices()
            .expect_err("missing pages should fail without default index");
        assert!(matches!(
            error,
            ChunkArrayCodecError::MissingPageWithoutDefault { .. }
        ));
    }

    #[test]
    fn chunk_array_duplicate_page_is_rejected() {
        let bounds = Aabb4i::new([0, 0, 0, 0], [3, 3, 3, 3]);
        let cell_count = bounds.chunk_cell_count().expect("valid bounds");
        let indices = vec![1u16; cell_count];

        let mut chunk_array = ChunkArrayData::from_dense_indices(
            bounds,
            vec![ChunkPayload::Empty, ChunkPayload::Uniform(1)],
            indices,
            Some(0),
        )
        .expect("encoding chunk array");

        // Duplicate the first encoded page by increasing page count and repeating payload bytes.
        let mut tampered = chunk_array.index_data.clone();
        let page_count =
            u32::from_le_bytes([tampered[20], tampered[21], tampered[22], tampered[23]]);
        let first_page_len = tampered.len() - 24;
        let first_page_payload = tampered[24..24 + first_page_len].to_vec();
        let new_count = page_count + 1;
        tampered[20..24].copy_from_slice(&new_count.to_le_bytes());
        tampered.extend_from_slice(&first_page_payload);
        chunk_array.index_data = tampered;

        let error = chunk_array
            .decode_dense_indices()
            .expect_err("duplicate page should be rejected");
        assert!(matches!(
            error,
            ChunkArrayCodecError::DuplicatePageIndex { .. }
        ));
    }

    #[test]
    fn chunk_array_default_index_is_validated() {
        let bounds = Aabb4i::new([0, 0, 0, 0], [0, 0, 0, 0]);
        let error =
            ChunkArrayData::from_dense_indices(bounds, vec![ChunkPayload::Empty], vec![0], Some(3))
                .expect_err("default index out of range should fail");

        assert!(matches!(
            error,
            ChunkArrayCodecError::DefaultIndexOutOfRange { .. }
        ));
    }

    #[test]
    fn region_chunk_tree_randomized_set_remove_matches_reference_model() {
        let domain = Aabb4i::new([-12, -4, -12, -12], [12, 4, 12, 12]);
        for seed in [0x31_u64, 0x7265_6769_6f6e_31, 0x7265_6769_6f6e_32] {
            let mut rng = DeterministicRng::new(seed);
            let mut tree = RegionChunkTree::new();
            let mut reference = ReferenceChunkStore::new();

            for step in 0..900 {
                let key = random_chunk_key_in_bounds(&mut rng, domain);
                let payload = if rng.chance(1, 4) {
                    None
                } else {
                    Some(random_chunk_payload(&mut rng))
                };

                let tree_changed = tree.set_chunk(key, payload.clone());
                let reference_changed = reference.set_chunk(key, payload);
                assert_eq!(
                    tree_changed, reference_changed,
                    "set/remove change mismatch seed={seed} step={step}"
                );

                if step % 75 == 0 {
                    assert_tree_matches_reference(&tree, &reference);

                    let probe_bounds = random_sub_bounds(&mut rng, domain);
                    let mut tree_subset = tree.collect_chunks_in_bounds(probe_bounds);
                    tree_subset.sort_unstable_by_key(|(chunk_key, _)| chunk_key.pos);
                    assert_eq!(
                        tree_subset,
                        reference.collect_chunks_in_bounds_sorted(probe_bounds),
                        "subset mismatch seed={seed} step={step}"
                    );
                    assert_eq!(
                        tree.any_non_empty_chunk_in_bounds(probe_bounds),
                        reference.any_non_empty_chunk_in_bounds(probe_bounds),
                        "non-empty mismatch seed={seed} step={step}"
                    );
                }
            }

            assert_tree_matches_reference(&tree, &reference);
        }
    }
}
