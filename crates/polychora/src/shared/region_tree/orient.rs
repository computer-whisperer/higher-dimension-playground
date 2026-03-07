use super::{RegionNodeKind, RegionTreeCore};
use crate::shared::chunk_payload::{ChunkArrayData, ChunkPayload};
use crate::shared::spatial::Aabb4i;
use crate::shared::voxel::{
    linear_cell_index, BlockData, TesseractOrientation, CHUNK_SIZE, CHUNK_VOLUME,
};

impl RegionTreeCore {
    /// Return a new tree with all voxel positions transformed by the given orientation.
    ///
    /// Rasterizes the tree contents, applies orientation to each block's lattice
    /// position, and rebuilds a new `ChunkArrayData` tree at the same scale.
    /// Suitable for small trees (blueprints, structures).
    pub fn oriented(&self, orientation: TesseractOrientation) -> Self {
        if orientation == TesseractOrientation::IDENTITY {
            return self.clone();
        }

        // Determine the scale of this tree. For mixed-scale Branch trees this
        // would need more sophistication, but blueprints are single ChunkArrays.
        let scale_exp = tree_scale_exp(self);

        // Collect all non-air, non-virgin blocks with their scaled-lattice positions.
        let mut blocks: Vec<([i64; 4], BlockData)> = Vec::new();
        collect_lattice_blocks(&self.kind, self.bounds, scale_exp, &mut blocks);

        if blocks.is_empty() {
            return RegionTreeCore {
                bounds: self.bounds,
                kind: RegionNodeKind::Empty,
                generator_version_hash: self.generator_version_hash,
            };
        }

        // Apply orientation to each lattice position.
        for (pos, _) in &mut blocks {
            let p = [pos[0] as i32, pos[1] as i32, pos[2] as i32, pos[3] as i32];
            let r = orientation.apply(p);
            *pos = [r[0] as i64, r[1] as i64, r[2] as i64, r[3] as i64];
        }

        // Compute chunk-aligned bounds in lattice space.
        let cs = CHUNK_SIZE as i64;
        let mut lat_min = blocks[0].0;
        let mut lat_max = blocks[0].0;
        for (pos, _) in &blocks {
            for i in 0..4 {
                lat_min[i] = lat_min[i].min(pos[i]);
                lat_max[i] = lat_max[i].max(pos[i]);
            }
        }

        let chunk_min: [i32; 4] = std::array::from_fn(|i| lat_min[i].div_euclid(cs) as i32);
        let chunk_max: [i32; 4] = std::array::from_fn(|i| lat_max[i].div_euclid(cs) as i32);

        let bounds = Aabb4i::from_lattice_bounds(chunk_min, chunk_max, scale_exp);
        let dims: [usize; 4] = std::array::from_fn(|i| (chunk_max[i] - chunk_min[i] + 1) as usize);
        let total_chunks = dims[0] * dims[1] * dims[2] * dims[3];

        // Build block palette and rasterize into per-chunk voxel arrays.
        let mut block_palette = vec![BlockData::AIR];
        let mut chunk_data: Vec<Vec<u16>> = (0..total_chunks)
            .map(|_| vec![0u16; CHUNK_VOLUME])
            .collect();

        for (pos, block) in &blocks {
            let palette_idx = intern_block(&mut block_palette, block);
            if palette_idx == 0 {
                continue;
            }

            let cx = (pos[0].div_euclid(cs) as i32 - chunk_min[0]) as usize;
            let cy = (pos[1].div_euclid(cs) as i32 - chunk_min[1]) as usize;
            let cz = (pos[2].div_euclid(cs) as i32 - chunk_min[2]) as usize;
            let cw = (pos[3].div_euclid(cs) as i32 - chunk_min[3]) as usize;

            let chunk_idx = cx + dims[0] * (cy + dims[1] * (cz + dims[2] * cw));

            let lx = pos[0].rem_euclid(cs) as usize;
            let ly = pos[1].rem_euclid(cs) as usize;
            let lz = pos[2].rem_euclid(cs) as usize;
            let lw = pos[3].rem_euclid(cs) as usize;
            let voxel_idx = linear_cell_index([lx, ly, lz, lw], [CHUNK_SIZE; 4]);

            chunk_data[chunk_idx][voxel_idx] = palette_idx;
        }

        // Build chunk palette, deduplicating empty chunks.
        let empty_chunk = vec![0u16; CHUNK_VOLUME];
        let mut chunk_palette: Vec<ChunkPayload> = vec![ChunkPayload::Empty];
        let mut dense_indices = Vec::with_capacity(total_chunks);

        for chunk in chunk_data {
            if chunk == empty_chunk {
                dense_indices.push(0u16);
                continue;
            }
            let found = chunk_palette.iter().position(|existing| {
                matches!(existing, ChunkPayload::Dense16 { materials } if materials == &chunk)
            });
            if let Some(idx) = found {
                dense_indices.push(idx as u16);
            } else {
                let idx = chunk_palette.len() as u16;
                chunk_palette.push(ChunkPayload::Dense16 { materials: chunk });
                dense_indices.push(idx);
            }
        }

        if dense_indices.iter().all(|&idx| idx == 0) {
            return RegionTreeCore {
                bounds,
                kind: RegionNodeKind::Empty,
                generator_version_hash: self.generator_version_hash,
            };
        }

        let chunk_array = ChunkArrayData::from_dense_indices_with_block_palette(
            bounds,
            chunk_palette,
            dense_indices,
            Some(0),
            block_palette,
            scale_exp,
        )
        .expect("oriented tree chunk array construction should not fail");

        RegionTreeCore {
            bounds,
            kind: RegionNodeKind::ChunkArray(chunk_array),
            generator_version_hash: self.generator_version_hash,
        }
    }
}

/// Extract the scale_exp from a tree. Returns 0 if no ChunkArrayData is found.
fn tree_scale_exp(tree: &RegionTreeCore) -> i8 {
    kind_scale_exp(&tree.kind)
}

fn kind_scale_exp(kind: &RegionNodeKind) -> i8 {
    match kind {
        RegionNodeKind::ChunkArray(ca) => ca.scale_exp,
        RegionNodeKind::Branch(children) => children
            .first()
            .map(|c| kind_scale_exp(&c.kind))
            .unwrap_or(0),
        _ => 0,
    }
}

/// Collect all non-air, non-virgin blocks with their scaled-lattice positions.
///
/// Positions are in the integer lattice at the given `scale_exp`:
/// `lattice_pos = world_pos / step(scale_exp)`.
fn collect_lattice_blocks(
    kind: &RegionNodeKind,
    bounds: Aabb4i,
    scale_exp: i8,
    out: &mut Vec<([i64; 4], BlockData)>,
) {
    match kind {
        RegionNodeKind::Empty | RegionNodeKind::ProceduralRef(_) => {}
        RegionNodeKind::Uniform(block) => {
            if block.is_air() || block.is_virgin() {
                return;
            }
            let cs = CHUNK_SIZE as i64;
            let (lmin, lmax) = bounds.to_chunk_lattice_bounds(scale_exp);
            for lw in lmin[3]..=lmax[3] {
                for lz in lmin[2]..=lmax[2] {
                    for ly in lmin[1]..=lmax[1] {
                        for lx in lmin[0]..=lmax[0] {
                            for vw in 0..CHUNK_SIZE as i64 {
                                for vz in 0..CHUNK_SIZE as i64 {
                                    for vy in 0..CHUNK_SIZE as i64 {
                                        for vx in 0..CHUNK_SIZE as i64 {
                                            out.push((
                                                [
                                                    lx as i64 * cs + vx,
                                                    ly as i64 * cs + vy,
                                                    lz as i64 * cs + vz,
                                                    lw as i64 * cs + vw,
                                                ],
                                                block.clone(),
                                            ));
                                        }
                                    }
                                }
                            }
                        }
                    }
                }
            }
        }
        RegionNodeKind::ChunkArray(chunk_array) => {
            let se = chunk_array.scale_exp;
            let cs = CHUNK_SIZE as i64;
            let Ok(indices) = chunk_array.decode_dense_indices() else {
                return;
            };
            let Some(extents) = chunk_array.bounds.chunk_extents_at_scale(se) else {
                return;
            };
            let (ca_lmin, ca_lmax) = chunk_array.bounds.to_chunk_lattice_bounds(se);
            for lw in ca_lmin[3]..=ca_lmax[3] {
                for lz in ca_lmin[2]..=ca_lmax[2] {
                    for ly in ca_lmin[1]..=ca_lmax[1] {
                        for lx in ca_lmin[0]..=ca_lmax[0] {
                            let local = [
                                (lx - ca_lmin[0]) as usize,
                                (ly - ca_lmin[1]) as usize,
                                (lz - ca_lmin[2]) as usize,
                                (lw - ca_lmin[3]) as usize,
                            ];
                            let linear = linear_cell_index(local, extents);
                            let Some(&palette_idx) = indices.get(linear) else {
                                continue;
                            };
                            let Some(payload) = chunk_array.chunk_palette.get(palette_idx as usize)
                            else {
                                continue;
                            };
                            let resolved = crate::shared::chunk_payload::ResolvedChunkPayload {
                                payload: payload.clone(),
                                block_palette: chunk_array.block_palette.clone(),
                            };
                            if matches!(payload, crate::shared::chunk_payload::ChunkPayload::Virgin)
                            {
                                continue;
                            }
                            for vw in 0..CHUNK_SIZE {
                                for vz in 0..CHUNK_SIZE {
                                    for vy in 0..CHUNK_SIZE {
                                        for vx in 0..CHUNK_SIZE {
                                            let voxel_idx = linear_cell_index(
                                                [vx, vy, vz, vw],
                                                [CHUNK_SIZE; 4],
                                            );
                                            let block = resolved.block_at(voxel_idx);
                                            if block.is_air() || block.is_virgin() {
                                                continue;
                                            }
                                            out.push((
                                                [
                                                    lx as i64 * cs + vx as i64,
                                                    ly as i64 * cs + vy as i64,
                                                    lz as i64 * cs + vz as i64,
                                                    lw as i64 * cs + vw as i64,
                                                ],
                                                block,
                                            ));
                                        }
                                    }
                                }
                            }
                        }
                    }
                }
            }
        }
        RegionNodeKind::Branch(children) => {
            for child in children {
                collect_lattice_blocks(&child.kind, child.bounds, scale_exp, out);
            }
        }
    }
}

fn intern_block(palette: &mut Vec<BlockData>, block: &BlockData) -> u16 {
    if block.is_air() {
        return 0;
    }
    if let Some(idx) = palette.iter().position(|b| b == block) {
        return idx as u16;
    }
    let idx = palette.len() as u16;
    palette.push(block.clone());
    idx
}
