use crate::shared::spatial::Aabb4i;
use crate::shared::voxel::{BlockData, CHUNK_VOLUME};
use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::fmt;

pub const MAX_CHUNKARRAY_CELLS_PER_NODE: usize = 256;
pub const PAGED_RLE_PAGE_EDGE_CELLS: usize = 4;
const PAGED_SPARSE_RLE_VERSION: u8 = 1;

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
    pub fn from_dense_materials_compact(materials: &[u16]) -> Result<Self, ChunkPayloadError> {
        if materials.len() != CHUNK_VOLUME {
            return Err(ChunkPayloadError::DenseLengthMismatch {
                expected: CHUNK_VOLUME,
                actual: materials.len(),
            });
        }

        if materials.iter().all(|m| *m == 0) {
            return Ok(Self::Empty);
        }

        let first = materials[0];
        if materials.iter().all(|m| *m == first) {
            return Ok(Self::Uniform(first));
        }

        let mut palette = Vec::<u16>::new();
        let mut lookup = HashMap::<u16, u16>::new();
        let mut indices = Vec::<u16>::with_capacity(CHUNK_VOLUME);

        for &material in materials {
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

        Ok(Self::PalettePacked {
            palette,
            bit_width,
            packed_indices,
        })
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
    /// Maps local voxel IDs (u16 values inside ChunkPayload variants) to rich block data.
    /// `block_palette[0]` is AIR by convention.
    pub block_palette: Vec<BlockData>,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum ChunkArrayCodecError {
    InvalidBounds,
    CellCountOverflow,
    DenseIndexCountMismatch {
        expected_cells: usize,
        actual_indices: usize,
    },
    DenseIndexLengthMismatch {
        expected_bytes: usize,
        actual_bytes: usize,
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
        page_index: usize,
        page_count: usize,
    },
    DuplicatePageIndex {
        page_index: usize,
    },
    MissingPageWithoutDefault {
        page_index: usize,
    },
    ZeroRunLength,
    PageRunLengthMismatch {
        expected_cells: usize,
        decoded_cells: usize,
    },
    TrailingPagedSparseBytes,
}

impl fmt::Display for ChunkArrayCodecError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::InvalidBounds => write!(f, "invalid chunk-array bounds"),
            Self::CellCountOverflow => write!(f, "chunk-array cell count overflow"),
            Self::DenseIndexCountMismatch {
                expected_cells,
                actual_indices,
            } => write!(
                f,
                "dense index count mismatch: expected {expected_cells}, got {actual_indices}"
            ),
            Self::DenseIndexLengthMismatch {
                expected_bytes,
                actual_bytes,
            } => write!(
                f,
                "dense index byte length mismatch: expected {expected_bytes}, got {actual_bytes}"
            ),
            Self::DefaultIndexOutOfRange {
                default_index,
                palette_len,
            } => write!(
                f,
                "default chunk index {default_index} out of range for palette length {palette_len}"
            ),
            Self::PaletteIndexOutOfRange { index, palette_len } => write!(
                f,
                "palette index {index} out of range for palette length {palette_len}"
            ),
            Self::UnsupportedPagedSparseVersion { expected, actual } => write!(
                f,
                "unsupported paged sparse version: expected {expected}, got {actual}"
            ),
            Self::UnsupportedPagedSparsePageEdge { expected, actual } => write!(
                f,
                "unsupported paged sparse page edge: expected {expected}, got {actual}"
            ),
            Self::MalformedPagedSparseData => write!(f, "malformed paged sparse rle data"),
            Self::PageIndexOutOfRange {
                page_index,
                page_count,
            } => {
                write!(f, "page index {page_index} out of range for {page_count} pages")
            }
            Self::DuplicatePageIndex { page_index } => {
                write!(f, "duplicate page index {page_index} in paged sparse data")
            }
            Self::MissingPageWithoutDefault { page_index } => {
                write!(f, "missing page {page_index} without default chunk index")
            }
            Self::ZeroRunLength => write!(f, "paged sparse run length must be >= 1"),
            Self::PageRunLengthMismatch {
                expected_cells,
                decoded_cells,
            } => write!(
                f,
                "paged sparse run length mismatch: expected {expected_cells} cells, decoded {decoded_cells}"
            ),
            Self::TrailingPagedSparseBytes => write!(f, "trailing paged sparse bytes"),
        }
    }
}

impl std::error::Error for ChunkArrayCodecError {}

impl ChunkArrayData {
    pub fn from_dense_indices(
        bounds: Aabb4i,
        chunk_palette: Vec<ChunkPayload>,
        dense_indices: Vec<u16>,
        default_chunk_idx: Option<u16>,
    ) -> Result<Self, ChunkArrayCodecError> {
        Self::from_dense_indices_with_block_palette(
            bounds,
            chunk_palette,
            dense_indices,
            default_chunk_idx,
            vec![BlockData::AIR],
        )
    }

    pub fn from_dense_indices_with_block_palette(
        bounds: Aabb4i,
        chunk_palette: Vec<ChunkPayload>,
        dense_indices: Vec<u16>,
        default_chunk_idx: Option<u16>,
        block_palette: Vec<BlockData>,
    ) -> Result<Self, ChunkArrayCodecError> {
        let dims = bounds_extents(bounds)?;
        let expected_cells = dims_cell_count(dims)?;
        if dense_indices.len() != expected_cells {
            return Err(ChunkArrayCodecError::DenseIndexCountMismatch {
                expected_cells,
                actual_indices: dense_indices.len(),
            });
        }
        validate_default_index(default_chunk_idx, chunk_palette.len())?;
        validate_palette_indices(&dense_indices, chunk_palette.len())?;

        let index_data = encode_paged_sparse_rle(dims, &dense_indices, default_chunk_idx);

        Ok(Self {
            bounds,
            chunk_palette,
            index_codec: ChunkArrayIndexCodec::PagedSparseRle,
            index_data,
            default_chunk_idx,
            block_palette,
        })
    }

    /// Find or insert a BlockData in the block palette, return its index.
    pub fn intern_block(&mut self, block: &BlockData) -> u16 {
        if let Some(idx) = self.block_palette.iter().position(|b| b == block) {
            return idx as u16;
        }
        let idx = self.block_palette.len() as u16;
        self.block_palette.push(block.clone());
        idx
    }

    /// Remap all u16 voxel values in chunk payloads using a translation table.
    pub fn remap_block_indices(&mut self, remap: &[u16]) {
        for payload in &mut self.chunk_palette {
            match payload {
                ChunkPayload::Empty => {}
                ChunkPayload::Uniform(ref mut material) => {
                    if let Some(&new_val) = remap.get(*material as usize) {
                        *material = new_val;
                    }
                }
                ChunkPayload::Dense16 { ref mut materials } => {
                    for m in materials.iter_mut() {
                        if let Some(&new_val) = remap.get(*m as usize) {
                            *m = new_val;
                        }
                    }
                }
                ChunkPayload::PalettePacked {
                    ref mut palette, ..
                } => {
                    for p in palette.iter_mut() {
                        if let Some(&new_val) = remap.get(*p as usize) {
                            *p = new_val;
                        }
                    }
                }
            }
        }
    }

    pub fn decode_dense_indices(&self) -> Result<Vec<u16>, ChunkArrayCodecError> {
        if self.chunk_palette.is_empty() {
            return Ok(Vec::new());
        }
        validate_default_index(self.default_chunk_idx, self.chunk_palette.len())?;

        let dims = bounds_extents(self.bounds)?;
        match self.index_codec {
            ChunkArrayIndexCodec::DenseU16 => {
                let expected_bytes = dims_cell_count(dims)?
                    .checked_mul(2)
                    .ok_or(ChunkArrayCodecError::CellCountOverflow)?;
                if self.index_data.len() != expected_bytes {
                    return Err(ChunkArrayCodecError::DenseIndexLengthMismatch {
                        expected_bytes,
                        actual_bytes: self.index_data.len(),
                    });
                }
                let mut out = Vec::with_capacity(expected_bytes / 2);
                for bytes in self.index_data.chunks_exact(2) {
                    out.push(u16::from_le_bytes([bytes[0], bytes[1]]));
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
    let word_count = total_bits.div_ceil(64);
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
    let cells_per_page = edge.pow(4);

    let page_dims = [
        dims[0].div_ceil(edge),
        dims[1].div_ceil(edge),
        dims[2].div_ceil(edge),
        dims[3].div_ceil(edge),
    ];
    let page_count = page_dims[0] * page_dims[1] * page_dims[2] * page_dims[3];

    let mut page_indices = Vec::<(u32, Vec<u16>)>::new();
    for page_idx in 0..page_count {
        let pw = page_idx / (page_dims[0] * page_dims[1] * page_dims[2]);
        let rem_w = page_idx % (page_dims[0] * page_dims[1] * page_dims[2]);
        let pz = rem_w / (page_dims[0] * page_dims[1]);
        let rem_z = rem_w % (page_dims[0] * page_dims[1]);
        let py = rem_z / page_dims[0];
        let px = rem_z % page_dims[0];

        let mut page = Vec::with_capacity(cells_per_page);
        for lw in 0..edge {
            for lz in 0..edge {
                for ly in 0..edge {
                    for lx in 0..edge {
                        let x = px * edge + lx;
                        let y = py * edge + ly;
                        let z = pz * edge + lz;
                        let w = pw * edge + lw;
                        if x < dims[0] && y < dims[1] && z < dims[2] && w < dims[3] {
                            let linear = x + dims[0] * (y + dims[1] * (z + dims[2] * w));
                            page.push(indices[linear]);
                        } else {
                            page.push(default_idx.unwrap_or(0));
                        }
                    }
                }
            }
        }

        let page_is_default = match default_idx {
            Some(default) => page.iter().all(|idx| *idx == default),
            None => false,
        };

        if !page_is_default {
            page_indices.push((page_idx as u32, page));
        }
    }

    let mut out = Vec::new();
    out.push(PAGED_SPARSE_RLE_VERSION);
    out.push(PAGED_RLE_PAGE_EDGE_CELLS as u8);
    out.extend_from_slice(&(page_indices.len() as u32).to_le_bytes());

    for (page_idx, page) in page_indices {
        out.extend_from_slice(&page_idx.to_le_bytes());

        let mut runs = Vec::<(u16, u16)>::new();
        let mut current = page[0];
        let mut len = 1u16;
        for &idx in &page[1..] {
            if idx == current && len < u16::MAX {
                len += 1;
            } else {
                runs.push((current, len));
                current = idx;
                len = 1;
            }
        }
        runs.push((current, len));

        out.extend_from_slice(&(runs.len() as u16).to_le_bytes());
        for (value, run_len) in runs {
            out.extend_from_slice(&value.to_le_bytes());
            out.extend_from_slice(&run_len.to_le_bytes());
        }
    }

    out
}

fn decode_paged_sparse_rle(
    dims: [usize; 4],
    bytes: &[u8],
    default_idx: Option<u16>,
    palette_len: usize,
) -> Result<Vec<u16>, ChunkArrayCodecError> {
    let total_cells = dims_cell_count(dims)?;
    if bytes.len() < 6 {
        return Err(ChunkArrayCodecError::MalformedPagedSparseData);
    }

    let version = bytes[0];
    if version != PAGED_SPARSE_RLE_VERSION {
        return Err(ChunkArrayCodecError::UnsupportedPagedSparseVersion {
            expected: PAGED_SPARSE_RLE_VERSION,
            actual: version,
        });
    }

    let page_edge = bytes[1];
    if page_edge != PAGED_RLE_PAGE_EDGE_CELLS as u8 {
        return Err(ChunkArrayCodecError::UnsupportedPagedSparsePageEdge {
            expected: PAGED_RLE_PAGE_EDGE_CELLS as u8,
            actual: page_edge,
        });
    }

    let page_count_declared = u32::from_le_bytes([bytes[2], bytes[3], bytes[4], bytes[5]]) as usize;

    let edge = PAGED_RLE_PAGE_EDGE_CELLS;
    let cells_per_page = edge.pow(4);
    let page_dims = [
        dims[0].div_ceil(edge),
        dims[1].div_ceil(edge),
        dims[2].div_ceil(edge),
        dims[3].div_ceil(edge),
    ];
    let page_count = page_dims[0] * page_dims[1] * page_dims[2] * page_dims[3];

    let mut cursor = 6usize;
    let mut touched = vec![false; page_count];
    let mut out = match default_idx {
        Some(default) => vec![default; total_cells],
        None => vec![0u16; total_cells],
    };

    for _ in 0..page_count_declared {
        if cursor + 6 > bytes.len() {
            return Err(ChunkArrayCodecError::MalformedPagedSparseData);
        }

        let page_idx = u32::from_le_bytes([
            bytes[cursor],
            bytes[cursor + 1],
            bytes[cursor + 2],
            bytes[cursor + 3],
        ]);
        cursor += 4;

        let page_idx_usize =
            usize::try_from(page_idx).map_err(|_| ChunkArrayCodecError::PageIndexOutOfRange {
                page_index: usize::MAX,
                page_count,
            })?;
        if page_idx_usize >= page_count {
            return Err(ChunkArrayCodecError::PageIndexOutOfRange {
                page_index: page_idx_usize,
                page_count,
            });
        }
        if touched[page_idx_usize] {
            return Err(ChunkArrayCodecError::DuplicatePageIndex {
                page_index: page_idx_usize,
            });
        }
        touched[page_idx_usize] = true;

        let run_count = u16::from_le_bytes([bytes[cursor], bytes[cursor + 1]]) as usize;
        cursor += 2;

        let mut decoded_cells = 0usize;
        let mut page_values = Vec::<u16>::with_capacity(cells_per_page);
        for _ in 0..run_count {
            if cursor + 4 > bytes.len() {
                return Err(ChunkArrayCodecError::MalformedPagedSparseData);
            }
            let value = u16::from_le_bytes([bytes[cursor], bytes[cursor + 1]]);
            let run_len = u16::from_le_bytes([bytes[cursor + 2], bytes[cursor + 3]]) as usize;
            cursor += 4;

            if run_len == 0 {
                return Err(ChunkArrayCodecError::ZeroRunLength);
            }
            if usize::from(value) >= palette_len {
                return Err(ChunkArrayCodecError::PaletteIndexOutOfRange {
                    index: value,
                    palette_len,
                });
            }

            decoded_cells += run_len;
            if decoded_cells > cells_per_page {
                return Err(ChunkArrayCodecError::PageRunLengthMismatch {
                    expected_cells: cells_per_page,
                    decoded_cells,
                });
            }
            page_values.extend(std::iter::repeat_n(value, run_len));
        }

        if decoded_cells != cells_per_page {
            return Err(ChunkArrayCodecError::PageRunLengthMismatch {
                expected_cells: cells_per_page,
                decoded_cells,
            });
        }

        let pw = page_idx_usize / (page_dims[0] * page_dims[1] * page_dims[2]);
        let rem_w = page_idx_usize % (page_dims[0] * page_dims[1] * page_dims[2]);
        let pz = rem_w / (page_dims[0] * page_dims[1]);
        let rem_z = rem_w % (page_dims[0] * page_dims[1]);
        let py = rem_z / page_dims[0];
        let px = rem_z % page_dims[0];

        let mut local = 0usize;
        for lw in 0..edge {
            for lz in 0..edge {
                for ly in 0..edge {
                    for lx in 0..edge {
                        let x = px * edge + lx;
                        let y = py * edge + ly;
                        let z = pz * edge + lz;
                        let w = pw * edge + lw;
                        let value = page_values[local];
                        local += 1;
                        if x < dims[0] && y < dims[1] && z < dims[2] && w < dims[3] {
                            let linear = x + dims[0] * (y + dims[1] * (z + dims[2] * w));
                            out[linear] = value;
                        }
                    }
                }
            }
        }
    }

    if cursor != bytes.len() {
        return Err(ChunkArrayCodecError::TrailingPagedSparseBytes);
    }

    if default_idx.is_none() {
        for (idx, was_touched) in touched.iter().enumerate() {
            if !*was_touched {
                return Err(ChunkArrayCodecError::MissingPageWithoutDefault { page_index: idx });
            }
        }
    }

    Ok(out)
}

// ---------------------------------------------------------------------------
// ResolvedChunkPayload — pairs opaque storage with its block palette
// ---------------------------------------------------------------------------

/// A `ChunkPayload` together with the `block_palette` needed to interpret it.
///
/// The `u16` values inside `ChunkPayload` are opaque palette indices into
/// `block_palette`. This type is the public API boundary for reading and
/// writing chunks; raw `ChunkPayload` remains an internal storage detail.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct ResolvedChunkPayload {
    pub payload: ChunkPayload,
    pub block_palette: Vec<BlockData>,
}

impl ResolvedChunkPayload {
    /// A chunk uniformly filled with a single block type.
    pub fn uniform(block: BlockData) -> Self {
        if block.is_air() {
            return Self::empty();
        }
        Self {
            payload: ChunkPayload::Uniform(1),
            block_palette: vec![BlockData::AIR, block],
        }
    }

    /// An empty (all-air) chunk.
    pub fn empty() -> Self {
        Self {
            payload: ChunkPayload::Empty,
            block_palette: vec![BlockData::AIR],
        }
    }

    /// True if this chunk contains at least one non-air block.
    pub fn has_solid_block(&self) -> bool {
        match &self.payload {
            ChunkPayload::Empty => false,
            ChunkPayload::Uniform(idx) => self
                .block_palette
                .get(*idx as usize)
                .map(|b| !b.is_air())
                .unwrap_or(false),
            ChunkPayload::Dense16 { materials } => materials.iter().any(|idx| {
                self.block_palette
                    .get(*idx as usize)
                    .map(|b| !b.is_air())
                    .unwrap_or(false)
            }),
            ChunkPayload::PalettePacked { palette, .. } => palette.iter().any(|idx| {
                self.block_palette
                    .get(*idx as usize)
                    .map(|b| !b.is_air())
                    .unwrap_or(false)
            }),
        }
    }

    /// Resolve the block at a given voxel index (0..CHUNK_VOLUME).
    pub fn block_at(&self, voxel_idx: usize) -> BlockData {
        let palette_idx = match &self.payload {
            ChunkPayload::Empty => 0u16,
            ChunkPayload::Uniform(idx) => *idx,
            ChunkPayload::Dense16 { materials } => {
                materials.get(voxel_idx).copied().unwrap_or(0)
            }
            ChunkPayload::PalettePacked { .. } => {
                match self.payload.dense_materials() {
                    Ok(dense) => dense.get(voxel_idx).copied().unwrap_or(0),
                    Err(_) => 0,
                }
            }
        };
        self.block_palette
            .get(palette_idx as usize)
            .cloned()
            .unwrap_or(BlockData::AIR)
    }

    /// Expand all voxels to a dense `Vec<BlockData>` of length CHUNK_VOLUME.
    pub fn dense_blocks(&self) -> Vec<BlockData> {
        match self.payload.dense_materials() {
            Ok(indices) => indices
                .iter()
                .map(|idx| {
                    self.block_palette
                        .get(*idx as usize)
                        .cloned()
                        .unwrap_or(BlockData::AIR)
                })
                .collect(),
            Err(_) => vec![BlockData::AIR; CHUNK_VOLUME],
        }
    }

    /// If the payload is uniform, return the resolved block.
    pub fn uniform_block(&self) -> Option<&BlockData> {
        match &self.payload {
            ChunkPayload::Uniform(idx) => self.block_palette.get(*idx as usize),
            ChunkPayload::Empty => self.block_palette.first(),
            _ => None,
        }
    }

    /// Build from a legacy `ChunkPayload` where u16 values are material IDs
    /// (i.e. `block_type` values with namespace=0).
    pub fn from_legacy_payload(payload: ChunkPayload) -> Self {
        match &payload {
            ChunkPayload::Empty => Self::empty(),
            ChunkPayload::Uniform(material) => {
                if *material == 0 {
                    Self::empty()
                } else {
                    Self::uniform(BlockData::simple(0, *material as u32))
                }
            }
            _ => {
                let Ok(materials) = payload.dense_materials() else {
                    return Self::empty();
                };
                let blocks: Vec<BlockData> = materials
                    .iter()
                    .map(|&m| {
                        if m == 0 {
                            BlockData::AIR
                        } else {
                            BlockData::simple(0, m as u32)
                        }
                    })
                    .collect();
                Self::from_dense_blocks(&blocks).unwrap_or_else(|_| Self::empty())
            }
        }
    }

    /// Convert to a legacy `ChunkPayload` where u16 values are material IDs.
    /// Uses `block_to_material_fn` to map each BlockData to a u8 material appearance.
    pub fn to_legacy_payload(&self, block_to_material_fn: impl Fn(&BlockData) -> u8) -> ChunkPayload {
        match &self.payload {
            ChunkPayload::Empty => ChunkPayload::Empty,
            ChunkPayload::Uniform(idx) => {
                let block = self
                    .block_palette
                    .get(*idx as usize)
                    .cloned()
                    .unwrap_or(BlockData::AIR);
                let mat = block_to_material_fn(&block);
                ChunkPayload::Uniform(mat as u16)
            }
            _ => {
                let Ok(palette_indices) = self.payload.dense_materials() else {
                    return ChunkPayload::Empty;
                };
                let materials: Vec<u16> = palette_indices
                    .iter()
                    .map(|&idx| {
                        let block = self
                            .block_palette
                            .get(idx as usize)
                            .cloned()
                            .unwrap_or(BlockData::AIR);
                        block_to_material_fn(&block) as u16
                    })
                    .collect();
                ChunkPayload::from_dense_materials_compact(&materials)
                    .unwrap_or(ChunkPayload::Dense16 { materials })
            }
        }
    }

    /// Build from a dense array of BlockData (length must be CHUNK_VOLUME).
    pub fn from_dense_blocks(blocks: &[BlockData]) -> Result<Self, ChunkPayloadError> {
        if blocks.len() != CHUNK_VOLUME {
            return Err(ChunkPayloadError::DenseLengthMismatch {
                expected: CHUNK_VOLUME,
                actual: blocks.len(),
            });
        }

        // Build block palette and map blocks to palette indices.
        let mut block_palette = vec![BlockData::AIR];
        let mut block_to_idx = HashMap::<BlockData, u16>::new();
        block_to_idx.insert(BlockData::AIR, 0);

        let mut dense_indices = Vec::with_capacity(CHUNK_VOLUME);
        for block in blocks {
            let idx = match block_to_idx.get(block) {
                Some(&idx) => idx,
                None => {
                    let idx = block_palette.len() as u16;
                    block_palette.push(block.clone());
                    block_to_idx.insert(block.clone(), idx);
                    idx
                }
            };
            dense_indices.push(idx);
        }

        let payload = ChunkPayload::from_dense_materials_compact(&dense_indices)?;
        Ok(Self {
            payload,
            block_palette,
        })
    }
}
