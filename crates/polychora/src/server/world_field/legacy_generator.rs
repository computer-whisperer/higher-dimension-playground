use super::{QueryDetail, QueryVolume, WorldField};
use crate::server::procgen;
use crate::shared::chunk_payload::{ChunkArrayData, ChunkPayload};
use crate::shared::region_tree::{RegionNodeKind, RegionTreeCore};
use crate::shared::spatial::Aabb4i;
use crate::shared::voxel::{BaseWorldKind, ChunkPos, VoxelType, CHUNK_SIZE, CHUNK_VOLUME};
use std::collections::{BTreeSet, HashMap, HashSet};
use std::sync::Arc;

const FLAT_FLOOR_CHUNK_Y: i32 = -1;
type DenseChunk = [VoxelType; CHUNK_VOLUME];

#[derive(Debug)]
pub struct LegacyWorldGenerator {
    base_kind: BaseWorldKind,
    flat_floor_chunk: DenseChunk,
    world_seed: u64,
    procgen_structures: bool,
    blocked_cells: HashSet<procgen::StructureCell>,
}

impl LegacyWorldGenerator {
    pub fn from_chunk_payloads(
        base_kind: BaseWorldKind,
        _chunk_payloads: impl IntoIterator<Item = ([i32; 4], ChunkPayload)>,
        world_seed: u64,
        procgen_structures: bool,
        blocked_cells: HashSet<procgen::StructureCell>,
    ) -> Self {
        let flat_floor_chunk = match base_kind {
            BaseWorldKind::FlatFloor { material } => build_flat_floor_chunk(material),
            BaseWorldKind::Empty => empty_dense_chunk(),
        };
        Self {
            base_kind,
            flat_floor_chunk,
            world_seed,
            procgen_structures,
            blocked_cells,
        }
    }

    pub fn set_procgen_blocked_cells(&mut self, blocked_cells: HashSet<procgen::StructureCell>) {
        self.blocked_cells = blocked_cells;
    }

    fn procgen_keepout_cells(&self) -> Option<&HashSet<procgen::StructureCell>> {
        if self.blocked_cells.is_empty() {
            None
        } else {
            Some(&self.blocked_cells)
        }
    }

    fn clone_base_chunk_or_empty_at(&self, chunk_pos: ChunkPos) -> DenseChunk {
        match self.base_kind {
            BaseWorldKind::Empty => empty_dense_chunk(),
            BaseWorldKind::FlatFloor { .. } if chunk_pos.y == FLAT_FLOOR_CHUNK_Y => {
                self.flat_floor_chunk
            }
            BaseWorldKind::FlatFloor { .. } => empty_dense_chunk(),
        }
    }

    fn build_virgin_chunk(&self, chunk_pos: ChunkPos) -> Option<DenseChunk> {
        let mut chunk = self.clone_base_chunk_or_empty_at(chunk_pos);
        if self.procgen_structures
            && procgen::structure_chunk_has_content_with_keepout(
                self.world_seed,
                chunk_pos,
                self.procgen_keepout_cells(),
            )
        {
            if let Some(structure_chunk) = procgen::generate_structure_chunk_with_keepout(
                self.world_seed,
                chunk_pos,
                self.procgen_keepout_cells(),
            ) {
                merge_non_air_voxels(&mut chunk, &structure_chunk);
            }
        }
        (!dense_chunk_is_empty(&chunk)).then_some(chunk)
    }

    fn base_default_payload_for_chunk_y(&self, chunk_y: i32) -> ChunkPayload {
        match self.base_kind {
            BaseWorldKind::Empty => ChunkPayload::Empty,
            BaseWorldKind::FlatFloor { .. }
                if chunk_y == FLAT_FLOOR_CHUNK_Y
                    && !dense_chunk_is_empty(&self.flat_floor_chunk) =>
            {
                payload_from_chunk_dense(&self.flat_floor_chunk)
            }
            BaseWorldKind::FlatFloor { .. } => ChunkPayload::Empty,
        }
    }

    fn query_candidate_chunk_ys(&self, bounds: Aabb4i) -> Vec<i32> {
        if !bounds.is_valid() {
            return Vec::new();
        }

        let mut ys = BTreeSet::new();
        if let BaseWorldKind::FlatFloor { .. } = self.base_kind {
            if !dense_chunk_is_empty(&self.flat_floor_chunk)
                && bounds.min[1] <= FLAT_FLOOR_CHUNK_Y
                && bounds.max[1] >= FLAT_FLOOR_CHUNK_Y
            {
                ys.insert(FLAT_FLOOR_CHUNK_Y);
            }
        }

        if self.procgen_structures {
            let (procgen_min_y, procgen_max_y) = procgen::structure_chunk_y_bounds();
            let min_y = bounds.min[1].max(procgen_min_y);
            let max_y = bounds.max[1].min(procgen_max_y);
            if min_y <= max_y {
                for y in min_y..=max_y {
                    ys.insert(y);
                }
            }
        }

        ys.into_iter().collect()
    }

    fn build_sparse_query_slice_at_y(
        &self,
        bounds: Aabb4i,
        chunk_y: i32,
    ) -> Option<RegionTreeCore> {
        if !bounds.is_valid() || chunk_y < bounds.min[1] || chunk_y > bounds.max[1] {
            return None;
        }

        let slice_bounds = Aabb4i::new(
            [bounds.min[0], chunk_y, bounds.min[2], bounds.min[3]],
            [bounds.max[0], chunk_y, bounds.max[2], bounds.max[3]],
        );
        let extents = slice_bounds.chunk_extents()?;
        let x_extent = extents[0];
        let z_extent = extents[2];
        let w_extent = extents[3];
        let cell_count = x_extent.checked_mul(z_extent)?.checked_mul(w_extent)?;

        let default_payload = self.base_default_payload_for_chunk_y(chunk_y);
        if !self.procgen_structures {
            if payload_is_effectively_empty(&default_payload) {
                return None;
            }
            let chunk_array = ChunkArrayData::from_dense_indices(
                slice_bounds,
                vec![default_payload],
                vec![0u16; cell_count],
                Some(0),
            )
            .expect("query_region_core default slice encoding should be valid");
            return Some(RegionTreeCore {
                bounds: slice_bounds,
                kind: RegionNodeKind::ChunkArray(chunk_array),
                generator_version_hash: 0,
            });
        }

        let mut found_non_empty = false;
        let mut min_non_empty = [i32::MAX; 3];
        let mut max_non_empty = [i32::MIN; 3];

        let mut palette = vec![default_payload.clone()];
        let mut palette_lookup = HashMap::<ChunkPayload, u16>::new();
        palette_lookup.insert(default_payload.clone(), 0);
        let mut dense_indices = vec![0u16; cell_count];

        for (w_idx, w) in (slice_bounds.min[3]..=slice_bounds.max[3]).enumerate() {
            for (z_idx, z) in (slice_bounds.min[2]..=slice_bounds.max[2]).enumerate() {
                for (x_idx, x) in (slice_bounds.min[0]..=slice_bounds.max[0]).enumerate() {
                    let chunk_pos = ChunkPos::new(x, chunk_y, z, w);
                    let maybe_chunk = self.build_virgin_chunk(chunk_pos);
                    if maybe_chunk.is_some() {
                        if !found_non_empty {
                            min_non_empty = [x, z, w];
                            max_non_empty = [x, z, w];
                            found_non_empty = true;
                        } else {
                            min_non_empty[0] = min_non_empty[0].min(x);
                            min_non_empty[1] = min_non_empty[1].min(z);
                            min_non_empty[2] = min_non_empty[2].min(w);
                            max_non_empty[0] = max_non_empty[0].max(x);
                            max_non_empty[1] = max_non_empty[1].max(z);
                            max_non_empty[2] = max_non_empty[2].max(w);
                        }
                    }

                    let effective_payload = match maybe_chunk {
                        Some(chunk) => payload_from_chunk_dense(&chunk),
                        None => ChunkPayload::Empty,
                    };
                    if effective_payload == default_payload {
                        continue;
                    }

                    let palette_idx = match palette_lookup.get(&effective_payload) {
                        Some(idx) => *idx,
                        None => {
                            let next_idx = palette.len() as u16;
                            palette_lookup.insert(effective_payload.clone(), next_idx);
                            palette.push(effective_payload);
                            next_idx
                        }
                    };
                    let linear = x_idx + x_extent * (z_idx + z_extent * w_idx);
                    dense_indices[linear] = palette_idx;
                }
            }
        }

        if !found_non_empty {
            return None;
        }

        let trimmed_bounds = Aabb4i::new(
            [
                min_non_empty[0],
                chunk_y,
                min_non_empty[1],
                min_non_empty[2],
            ],
            [
                max_non_empty[0],
                chunk_y,
                max_non_empty[1],
                max_non_empty[2],
            ],
        );

        let indices = if trimmed_bounds == slice_bounds {
            dense_indices
        } else {
            let mut trimmed_indices = Vec::new();
            for w in trimmed_bounds.min[3]..=trimmed_bounds.max[3] {
                let full_w_idx = usize::try_from(w - slice_bounds.min[3]).ok()?;
                for z in trimmed_bounds.min[2]..=trimmed_bounds.max[2] {
                    let full_z_idx = usize::try_from(z - slice_bounds.min[2]).ok()?;
                    for x in trimmed_bounds.min[0]..=trimmed_bounds.max[0] {
                        let full_x_idx = usize::try_from(x - slice_bounds.min[0]).ok()?;
                        let linear = full_x_idx + x_extent * (full_z_idx + z_extent * full_w_idx);
                        trimmed_indices.push(dense_indices[linear]);
                    }
                }
            }
            trimmed_indices
        };

        let chunk_array =
            ChunkArrayData::from_dense_indices(trimmed_bounds, palette, indices, Some(0))
                .expect("query_region_core slice encoding should be valid");
        Some(RegionTreeCore {
            bounds: trimmed_bounds,
            kind: RegionNodeKind::ChunkArray(chunk_array),
            generator_version_hash: 0,
        })
    }

    pub fn query_region_core(
        &self,
        query: QueryVolume,
        _detail: QueryDetail,
    ) -> Arc<RegionTreeCore> {
        let bounds = query.bounds;
        if !bounds.is_valid() {
            return Arc::new(RegionTreeCore {
                bounds,
                kind: RegionNodeKind::Empty,
                generator_version_hash: 0,
            });
        }
        if bounds.min == bounds.max {
            let chunk_pos =
                ChunkPos::new(bounds.min[0], bounds.min[1], bounds.min[2], bounds.min[3]);
            let kind = self
                .build_virgin_chunk(chunk_pos)
                .map(|chunk| payload_from_chunk_compact(&chunk))
                .map(|payload| single_chunk_kind(bounds, payload))
                .unwrap_or(RegionNodeKind::Empty);
            return Arc::new(RegionTreeCore {
                bounds,
                kind,
                generator_version_hash: 0,
            });
        }

        let mut children = Vec::new();
        for y in self.query_candidate_chunk_ys(bounds) {
            if let Some(child) = self.build_sparse_query_slice_at_y(bounds, y) {
                children.push(child);
            }
        }

        if children.is_empty() {
            return Arc::new(RegionTreeCore {
                bounds,
                kind: RegionNodeKind::Empty,
                generator_version_hash: 0,
            });
        }
        if children.len() == 1 && children[0].bounds == bounds {
            return Arc::new(children.pop().expect("single child"));
        }

        Arc::new(RegionTreeCore {
            bounds,
            kind: RegionNodeKind::Branch(children),
            generator_version_hash: 0,
        })
    }
}

impl WorldField for LegacyWorldGenerator {
    fn query_region_core(&self, query: QueryVolume, detail: QueryDetail) -> Arc<RegionTreeCore> {
        LegacyWorldGenerator::query_region_core(self, query, detail)
    }
}

fn payload_from_chunk_dense(chunk: &DenseChunk) -> ChunkPayload {
    let materials: Vec<u16> = chunk.iter().map(|voxel| u16::from(voxel.0)).collect();
    ChunkPayload::Dense16 { materials }
}

fn payload_is_effectively_empty(payload: &ChunkPayload) -> bool {
    match payload {
        ChunkPayload::Empty => true,
        ChunkPayload::Uniform(material) => *material == 0,
        ChunkPayload::Dense16 { materials } => materials.iter().all(|material| *material == 0),
        ChunkPayload::PalettePacked { .. } => payload
            .dense_materials()
            .map(|materials| materials.into_iter().all(|material| material == 0))
            .unwrap_or(false),
    }
}

fn single_chunk_kind(bounds: Aabb4i, payload: ChunkPayload) -> RegionNodeKind {
    match payload {
        ChunkPayload::Uniform(material) => {
            if material == 0 {
                RegionNodeKind::Empty
            } else {
                RegionNodeKind::Uniform(material)
            }
        }
        ChunkPayload::Empty => RegionNodeKind::Empty,
        other => ChunkArrayData::from_dense_indices(bounds, vec![other], vec![0u16], Some(0))
            .map(RegionNodeKind::ChunkArray)
            .unwrap_or(RegionNodeKind::Empty),
    }
}

fn payload_from_chunk_compact(chunk: &DenseChunk) -> ChunkPayload {
    let materials: Vec<u16> = chunk.iter().map(|voxel| u16::from(voxel.0)).collect();
    ChunkPayload::from_dense_materials_compact(&materials)
        .unwrap_or(ChunkPayload::Dense16 { materials })
}

fn build_flat_floor_chunk(material: VoxelType) -> DenseChunk {
    let mut chunk = empty_dense_chunk();
    if material.is_air() {
        return chunk;
    }
    let local_y_top = CHUNK_SIZE - 1;
    let local_y_bottom = CHUNK_SIZE - 2;
    for x in 0..CHUNK_SIZE {
        for z in 0..CHUNK_SIZE {
            for w in 0..CHUNK_SIZE {
                dense_chunk_set(&mut chunk, x, local_y_top, z, w, material);
                dense_chunk_set(&mut chunk, x, local_y_bottom, z, w, material);
            }
        }
    }
    chunk
}

fn merge_non_air_voxels(dst: &mut DenseChunk, src: &DenseChunk) {
    for (idx, voxel) in src.iter().enumerate() {
        if voxel.is_air() {
            continue;
        }
        dst[idx] = *voxel;
    }
}

fn empty_dense_chunk() -> DenseChunk {
    [VoxelType::AIR; CHUNK_VOLUME]
}

fn dense_chunk_is_empty(chunk: &DenseChunk) -> bool {
    chunk.iter().all(|voxel| voxel.is_air())
}

fn dense_chunk_set(
    chunk: &mut DenseChunk,
    x: usize,
    y: usize,
    z: usize,
    w: usize,
    voxel: VoxelType,
) {
    let idx =
        w * CHUNK_SIZE * CHUNK_SIZE * CHUNK_SIZE + z * CHUNK_SIZE * CHUNK_SIZE + y * CHUNK_SIZE + x;
    chunk[idx] = voxel;
}
