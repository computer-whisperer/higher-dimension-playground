use super::{QueryDetail, QueryVolume, WorldField};
use crate::server::procgen;
use crate::shared::chunk_payload::{ChunkArrayData, ChunkPayload};
use crate::shared::region_tree::{
    chunk_key_from_chunk_pos, chunk_pos_from_chunk_key, RegionChunkTree, RegionNodeKind,
    RegionTreeCore,
};
use crate::shared::spatial::Aabb4i;
use crate::shared::voxel::{
    world_to_chunk, BaseWorldKind, ChunkPos, VoxelType, CHUNK_SIZE, CHUNK_VOLUME,
};
use std::collections::{BTreeSet, HashMap, HashSet};
use std::sync::Arc;

const FLAT_FLOOR_CHUNK_Y: i32 = -1;
type DenseChunk = [VoxelType; CHUNK_VOLUME];

#[derive(Debug)]
pub struct LegacyWorldGenerator {
    base_kind: BaseWorldKind,
    flat_floor_chunk: DenseChunk,
    chunk_tree: RegionChunkTree,
    world_seed: u64,
    procgen_structures: bool,
    blocked_cells: HashSet<procgen::StructureCell>,
    world_dirty: bool,
}

impl LegacyWorldGenerator {
    pub fn from_chunk_payloads(
        base_kind: BaseWorldKind,
        chunk_payloads: impl IntoIterator<Item = ([i32; 4], ChunkPayload)>,
        world_seed: u64,
        procgen_structures: bool,
        blocked_cells: HashSet<procgen::StructureCell>,
    ) -> Self {
        let chunk_tree = RegionChunkTree::from_chunks(
            chunk_payloads
                .into_iter()
                .map(|(chunk_pos, payload)| (chunk_pos, payload)),
        );
        Self::from_chunk_tree(
            base_kind,
            chunk_tree,
            world_seed,
            procgen_structures,
            blocked_cells,
        )
    }

    pub fn from_chunk_tree(
        base_kind: BaseWorldKind,
        chunk_tree: RegionChunkTree,
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
            chunk_tree,
            world_seed,
            procgen_structures,
            blocked_cells,
            world_dirty: false,
        }
    }

    pub fn world_seed(&self) -> u64 {
        self.world_seed
    }

    pub fn base_kind(&self) -> BaseWorldKind {
        self.base_kind
    }

    pub fn chunk_tree(&self) -> &RegionChunkTree {
        &self.chunk_tree
    }

    pub fn any_dirty(&self) -> bool {
        self.world_dirty
    }

    pub fn clear_dirty(&mut self) {
        self.world_dirty = false;
    }

    pub fn non_empty_chunk_count(&self) -> usize {
        self.chunk_tree.non_empty_chunk_count()
    }

    pub fn set_procgen_blocked_cells(&mut self, blocked_cells: HashSet<procgen::StructureCell>) {
        self.blocked_cells = blocked_cells;
    }

    pub fn rebuild_procgen_keepout_from_chunks(&mut self, padding_chunks: i32) -> usize {
        if !self.procgen_structures {
            self.set_procgen_blocked_cells(HashSet::new());
            return 0;
        }

        let keepout_chunks = self.gather_chunks_with_padding(padding_chunks.max(0));
        let mut blocked = HashSet::new();
        for chunk_pos in keepout_chunks {
            for cell in procgen::structure_cells_affecting_chunk(self.world_seed, chunk_pos) {
                blocked.insert(cell);
            }
        }

        let blocked_count = blocked.len();
        self.set_procgen_blocked_cells(blocked);
        blocked_count
    }

    fn gather_chunks_with_padding(&self, padding_chunks: i32) -> HashSet<ChunkPos> {
        let padding = padding_chunks.max(0);
        let mut out = HashSet::new();
        for (pos, chunk) in self.collect_chunks() {
            if dense_chunk_is_empty(&chunk) {
                continue;
            }
            for dx in -padding..=padding {
                for dy in -padding..=padding {
                    for dz in -padding..=padding {
                        for dw in -padding..=padding {
                            out.insert(ChunkPos::new(
                                pos.x + dx,
                                pos.y + dy,
                                pos.z + dz,
                                pos.w + dw,
                            ));
                        }
                    }
                }
            }
        }
        out
    }

    pub fn has_chunk(&self, chunk_pos: ChunkPos) -> bool {
        self.chunk_tree
            .has_chunk(chunk_key_from_chunk_pos(chunk_pos))
    }

    pub fn chunk_at(&self, chunk_pos: ChunkPos) -> Option<DenseChunk> {
        self.chunk_tree
            .chunk_payload(chunk_key_from_chunk_pos(chunk_pos))
            .and_then(|payload| chunk_from_payload(&payload))
    }

    pub fn set_chunk(&mut self, chunk_pos: ChunkPos, chunk: DenseChunk) -> bool {
        let payload = payload_from_chunk_compact(&chunk);
        let changed = self
            .chunk_tree
            .set_chunk(chunk_key_from_chunk_pos(chunk_pos), Some(payload));
        if changed {
            self.world_dirty = true;
        }
        changed
    }

    pub fn remove_chunk(&mut self, chunk_pos: ChunkPos) -> bool {
        let changed = self
            .chunk_tree
            .remove_chunk(chunk_key_from_chunk_pos(chunk_pos));
        if changed {
            self.world_dirty = true;
        }
        changed
    }

    pub fn apply_chunk_payloads<I>(&mut self, chunk_payloads: I) -> usize
    where
        I: IntoIterator<Item = ([i32; 4], ChunkPayload)>,
    {
        let mut changed = 0usize;
        for (chunk_pos, payload) in chunk_payloads {
            if self.chunk_tree.set_chunk(chunk_pos, Some(payload)) {
                changed = changed.saturating_add(1);
            }
        }
        changed
    }

    pub fn any_non_empty_chunk_in_bounds(&self, bounds: Aabb4i) -> bool {
        self.chunk_tree.any_non_empty_chunk_in_bounds(bounds)
    }

    pub fn collect_chunks(&self) -> Vec<(ChunkPos, DenseChunk)> {
        self.chunk_tree
            .collect_chunks()
            .into_iter()
            .filter_map(|(key, payload)| {
                chunk_from_payload(&payload).map(|chunk| (chunk_pos_from_chunk_key(key), chunk))
            })
            .collect()
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

    fn build_virgin_chunk_for_edit(&self, chunk_pos: ChunkPos) -> Option<DenseChunk> {
        let mut chunk = self.clone_base_chunk_or_empty_at(chunk_pos);
        if self.procgen_structures {
            if procgen::structure_chunk_has_content_with_keepout(
                self.world_seed,
                chunk_pos,
                self.procgen_keepout_cells(),
            ) {
                if let Some(structure_chunk) = procgen::generate_structure_chunk_with_keepout(
                    self.world_seed,
                    chunk_pos,
                    self.procgen_keepout_cells(),
                ) {
                    merge_non_air_voxels(&mut chunk, &structure_chunk);
                }
            }
        }
        (!dense_chunk_is_empty(&chunk)).then_some(chunk)
    }

    pub fn effective_chunk(
        &self,
        chunk_pos: ChunkPos,
        preserve_explicit_empty_chunk: bool,
    ) -> Option<DenseChunk> {
        if let Some(chunk_data) = self.chunk_at(chunk_pos) {
            if preserve_explicit_empty_chunk {
                return Some(chunk_data);
            }
            return (!dense_chunk_is_empty(&chunk_data)).then_some(chunk_data);
        }
        self.build_virgin_chunk_for_edit(chunk_pos)
    }

    pub fn trim_chunk_if_virgin(&mut self, chunk_pos: ChunkPos) -> bool {
        let Some(chunk_data) = self.chunk_at(chunk_pos) else {
            return false;
        };

        let virgin_chunk = self.build_virgin_chunk_for_edit(chunk_pos);
        let matches_virgin = match virgin_chunk {
            Some(virgin) => chunks_equal(&chunk_data, &virgin),
            None => dense_chunk_is_empty(&chunk_data),
        };
        if matches_virgin {
            return self.remove_chunk(chunk_pos);
        }
        false
    }

    pub fn apply_voxel_edit(
        &mut self,
        position: [i32; 4],
        material: VoxelType,
    ) -> Option<ChunkPos> {
        let (chunk_pos, voxel_index) =
            world_to_chunk(position[0], position[1], position[2], position[3]);

        let existing_chunk = self.chunk_at(chunk_pos);
        let mut virgin_chunk_for_materialize = None;

        let current_voxel = if let Some(chunk_data) = existing_chunk.as_ref() {
            chunk_data[voxel_index]
        } else {
            virgin_chunk_for_materialize = self.build_virgin_chunk_for_edit(chunk_pos);
            virgin_chunk_for_materialize
                .as_ref()
                .map(|chunk| chunk[voxel_index])
                .unwrap_or(VoxelType::AIR)
        };

        if current_voxel == material {
            return None;
        }

        let mut chunk_data = existing_chunk
            .or(virgin_chunk_for_materialize)
            .unwrap_or_else(empty_dense_chunk);
        set_chunk_voxel_by_index(&mut chunk_data, voxel_index, material);
        let _ = self.set_chunk(chunk_pos, chunk_data);
        let _ = self.trim_chunk_if_virgin(chunk_pos);
        Some(chunk_pos)
    }

    pub fn prune_virgin_chunks(&mut self) -> usize {
        let positions: Vec<ChunkPos> = self
            .collect_chunks()
            .into_iter()
            .map(|(pos, _)| pos)
            .collect();

        let mut pruned = 0usize;
        for chunk_pos in positions {
            if self.trim_chunk_if_virgin(chunk_pos) {
                pruned = pruned.saturating_add(1);
            }
        }
        pruned
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

        for (key, _) in self.chunk_tree.collect_chunks_in_bounds(bounds) {
            ys.insert(key[1]);
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
                    let maybe_chunk = self.effective_chunk(chunk_pos, false);
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

fn payload_from_chunk_compact(chunk: &DenseChunk) -> ChunkPayload {
    let materials: Vec<u16> = chunk.iter().map(|voxel| u16::from(voxel.0)).collect();
    ChunkPayload::from_dense_materials_compact(&materials)
        .unwrap_or(ChunkPayload::Dense16 { materials })
}

fn chunk_from_payload(payload: &ChunkPayload) -> Option<DenseChunk> {
    let materials = payload.dense_materials().ok()?;
    if materials.len() != CHUNK_VOLUME {
        return None;
    }
    let mut chunk = empty_dense_chunk();
    for (idx, material) in materials.into_iter().enumerate() {
        chunk[idx] = VoxelType(u8::try_from(material).unwrap_or(u8::MAX));
    }
    Some(chunk)
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

fn chunks_equal(a: &DenseChunk, b: &DenseChunk) -> bool {
    a[..] == b[..]
}

fn set_chunk_voxel_by_index(chunk: &mut DenseChunk, voxel_index: usize, value: VoxelType) {
    chunk[voxel_index] = value;
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
