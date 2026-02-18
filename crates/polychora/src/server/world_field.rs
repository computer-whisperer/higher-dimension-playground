use super::procgen;
use crate::shared::voxel::{world_to_chunk, BaseWorldKind, Chunk, ChunkPos, VoxelType, CHUNK_SIZE};
use crate::shared::worldfield::{
    Aabb4i, ChunkArrayData, ChunkKey, ChunkPayload, QueryDetail, QueryVolume, RealizeProfile,
    RegionChunkTree, RegionNodeKind, RegionTreeCore, WorldField,
    QUERY_CONTENT_BOUNDS_SCAN_BUDGET_CELLS,
};
use std::collections::{BTreeSet, HashMap, HashSet};
use std::sync::Arc;

const FLAT_FLOOR_CHUNK_Y: i32 = -1;

#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
struct RealizeCacheKey {
    chunk_key: ChunkKey,
    profile: RealizeProfile,
}

#[derive(Debug)]
pub struct ServerWorldField {
    base_kind: BaseWorldKind,
    flat_floor_chunk: Chunk,
    chunk_tree: RegionChunkTree,
    world_seed: u64,
    procgen_structures: bool,
    blocked_cells: HashSet<procgen::StructureCell>,
    world_dirty: bool,
    realize_cache: HashMap<RealizeCacheKey, ChunkPayload>,
}

impl ServerWorldField {
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
                .map(|(chunk_pos, payload)| (ChunkKey { pos: chunk_pos }, payload)),
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
            BaseWorldKind::Empty => Chunk::new(),
        };
        Self {
            base_kind,
            flat_floor_chunk,
            chunk_tree,
            world_seed,
            procgen_structures,
            blocked_cells,
            world_dirty: false,
            realize_cache: HashMap::new(),
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
        self.realize_cache.clear();
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
            if chunk.is_empty() {
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
            .has_chunk(ChunkKey::from_chunk_pos(chunk_pos))
    }

    pub fn chunk_at(&self, chunk_pos: ChunkPos) -> Option<Chunk> {
        self.chunk_tree
            .chunk_payload(ChunkKey::from_chunk_pos(chunk_pos))
            .and_then(|payload| payload.to_voxel_chunk().ok())
    }

    pub fn set_chunk(&mut self, chunk_pos: ChunkPos, chunk: Chunk) -> bool {
        let payload = ChunkPayload::from_chunk_compact(&chunk);
        let changed = self
            .chunk_tree
            .set_chunk(ChunkKey::from_chunk_pos(chunk_pos), Some(payload));
        if changed {
            self.world_dirty = true;
            self.realize_cache.clear();
        }
        changed
    }

    pub fn remove_chunk(&mut self, chunk_pos: ChunkPos) -> bool {
        let changed = self
            .chunk_tree
            .remove_chunk(ChunkKey::from_chunk_pos(chunk_pos));
        if changed {
            self.world_dirty = true;
            self.realize_cache.clear();
        }
        changed
    }

    pub fn apply_chunk_payloads<I>(&mut self, chunk_payloads: I) -> usize
    where
        I: IntoIterator<Item = ([i32; 4], ChunkPayload)>,
    {
        let mut changed = 0usize;
        for (chunk_pos, payload) in chunk_payloads {
            if self
                .chunk_tree
                .set_chunk(ChunkKey { pos: chunk_pos }, Some(payload))
            {
                changed = changed.saturating_add(1);
            }
        }
        if changed > 0 {
            self.realize_cache.clear();
        }
        changed
    }

    pub fn any_non_empty_chunk_in_bounds(&self, bounds: Aabb4i) -> bool {
        self.chunk_tree.any_non_empty_chunk_in_bounds(bounds)
    }

    pub fn collect_chunks(&self) -> Vec<(ChunkPos, Chunk)> {
        self.chunk_tree
            .collect_chunks()
            .into_iter()
            .filter_map(|(key, payload)| {
                payload
                    .to_voxel_chunk()
                    .ok()
                    .map(|chunk| (key.to_chunk_pos(), chunk))
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

    fn clone_base_chunk_or_empty_at(&self, chunk_pos: ChunkPos) -> Chunk {
        match self.base_kind {
            BaseWorldKind::Empty => Chunk::new(),
            BaseWorldKind::FlatFloor { .. } if chunk_pos.y == FLAT_FLOOR_CHUNK_Y => {
                self.flat_floor_chunk.clone()
            }
            BaseWorldKind::FlatFloor { .. } => Chunk::new(),
        }
    }

    fn build_virgin_chunk_for_edit(&self, chunk_pos: ChunkPos) -> Option<Chunk> {
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
        (!chunk.is_empty()).then_some(chunk)
    }

    pub fn effective_chunk(
        &self,
        chunk_pos: ChunkPos,
        preserve_explicit_empty_chunk: bool,
    ) -> Option<Chunk> {
        if let Some(chunk_data) = self.chunk_at(chunk_pos) {
            if preserve_explicit_empty_chunk {
                return Some(chunk_data);
            }
            return (!chunk_data.is_empty()).then_some(chunk_data);
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
            None => chunk_data.is_empty(),
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
            chunk_data.voxels[voxel_index]
        } else {
            virgin_chunk_for_materialize = self.build_virgin_chunk_for_edit(chunk_pos);
            virgin_chunk_for_materialize
                .as_ref()
                .map(|chunk| chunk.voxels[voxel_index])
                .unwrap_or(VoxelType::AIR)
        };

        if current_voxel == material {
            return None;
        }

        let mut chunk_data = existing_chunk
            .or(virgin_chunk_for_materialize)
            .unwrap_or_else(Chunk::new);
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

    fn base_chunk_y_bounds_for_scale(&self, chunk_scale: i32) -> Option<(i32, i32)> {
        let scale = chunk_scale.max(1);
        match self.base_kind {
            BaseWorldKind::Empty => None,
            BaseWorldKind::FlatFloor { .. } => {
                if self.flat_floor_chunk.is_empty() {
                    None
                } else {
                    let y = FLAT_FLOOR_CHUNK_Y.div_euclid(scale);
                    Some((y, y))
                }
            }
        }
    }

    fn might_have_content_conservatively(&self, bounds: Aabb4i) -> bool {
        if !bounds.is_valid() {
            return false;
        }

        if self.chunk_tree.any_non_empty_chunk_in_bounds(bounds) {
            return true;
        }

        if let Some((base_min_y, base_max_y)) = self.base_chunk_y_bounds_for_scale(1) {
            if bounds.max[1] >= base_min_y && bounds.min[1] <= base_max_y {
                return true;
            }
        }

        if self.procgen_structures {
            let (procgen_min_y, procgen_max_y) = procgen::structure_chunk_y_bounds();
            if bounds.max[1] >= procgen_min_y && bounds.min[1] <= procgen_max_y {
                return true;
            }
        }

        false
    }

    fn base_default_payload_for_chunk_y(&self, chunk_y: i32) -> ChunkPayload {
        match self.base_kind {
            BaseWorldKind::Empty => ChunkPayload::Empty,
            BaseWorldKind::FlatFloor { .. }
                if chunk_y == FLAT_FLOOR_CHUNK_Y && !self.flat_floor_chunk.is_empty() =>
            {
                ChunkPayload::from_chunk_dense(&self.flat_floor_chunk)
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
            if !self.flat_floor_chunk.is_empty()
                && bounds.min[1] <= FLAT_FLOOR_CHUNK_Y
                && bounds.max[1] >= FLAT_FLOOR_CHUNK_Y
            {
                ys.insert(FLAT_FLOOR_CHUNK_Y);
            }
        }

        for (key, _) in self.chunk_tree.collect_chunks_in_bounds(bounds) {
            ys.insert(key.pos[1]);
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
                        Some(chunk) => ChunkPayload::from_chunk_dense(&chunk),
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
}

impl WorldField for ServerWorldField {
    fn query_region_core(&self, query: QueryVolume, _detail: QueryDetail) -> Arc<RegionTreeCore> {
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

    fn query_content_bounds(&self, query: QueryVolume) -> Option<Aabb4i> {
        let bounds = query.bounds;
        if !bounds.is_valid() {
            return None;
        }

        let Some(cell_count) = bounds.chunk_cell_count() else {
            return None;
        };

        if cell_count > QUERY_CONTENT_BOUNDS_SCAN_BUDGET_CELLS {
            return self
                .might_have_content_conservatively(bounds)
                .then_some(bounds);
        }

        let mut min = [i32::MAX; 4];
        let mut max = [i32::MIN; 4];
        let mut found = false;

        for w in bounds.min[3]..=bounds.max[3] {
            for z in bounds.min[2]..=bounds.max[2] {
                for y in bounds.min[1]..=bounds.max[1] {
                    for x in bounds.min[0]..=bounds.max[0] {
                        let chunk_pos = ChunkPos::new(x, y, z, w);
                        if self.effective_chunk(chunk_pos, false).is_none() {
                            continue;
                        }

                        if !found {
                            min = [x, y, z, w];
                            max = [x, y, z, w];
                            found = true;
                        } else {
                            min[0] = min[0].min(x);
                            min[1] = min[1].min(y);
                            min[2] = min[2].min(z);
                            min[3] = min[3].min(w);
                            max[0] = max[0].max(x);
                            max[1] = max[1].max(y);
                            max[2] = max[2].max(z);
                            max[3] = max[3].max(w);
                        }
                    }
                }
            }
        }

        found.then_some(Aabb4i::new(min, max))
    }

    fn realize_chunk(&mut self, key: ChunkKey, profile: RealizeProfile) -> ChunkPayload {
        let cache_key = RealizeCacheKey {
            chunk_key: key,
            profile,
        };
        if let Some(payload) = self.realize_cache.get(&cache_key) {
            return payload.clone();
        }

        let chunk_pos = key.to_chunk_pos();
        let payload = if let Some(chunk) = self.effective_chunk(chunk_pos, true) {
            ChunkPayload::from_chunk_dense(&chunk)
        } else {
            ChunkPayload::Empty
        };

        self.realize_cache.insert(cache_key, payload.clone());
        payload
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

fn merge_non_air_voxels(dst: &mut Chunk, src: &Chunk) {
    for (idx, voxel) in src.voxels.iter().enumerate() {
        if voxel.is_air() {
            continue;
        }
        dst.voxels[idx] = *voxel;
    }
    dst.solid_count = dst.voxels.iter().filter(|v| v.is_solid()).count() as u32;
    dst.dirty = true;
}

fn chunks_equal(a: &Chunk, b: &Chunk) -> bool {
    a.solid_count == b.solid_count && a.voxels[..] == b.voxels[..]
}

fn set_chunk_voxel_by_index(chunk: &mut Chunk, voxel_index: usize, value: VoxelType) {
    let old = chunk.voxels[voxel_index];
    if old == value {
        return;
    }
    if old.is_solid() && value.is_air() {
        chunk.solid_count = chunk.solid_count.saturating_sub(1);
    } else if old.is_air() && value.is_solid() {
        chunk.solid_count = chunk.solid_count.saturating_add(1);
    }
    chunk.voxels[voxel_index] = value;
    chunk.dirty = true;
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::server::region_tree_cache::RegionTreeWorkingSet;
    use crate::shared::voxel::{world_to_chunk, BaseWorldKind, Chunk, RegionChunkWorld, VoxelType};
    use crate::shared::worldfield::{
        collect_non_empty_chunks_from_core_in_bounds, ChunkArrayIndexCodec,
        QUERY_CONTENT_BOUNDS_SCAN_BUDGET_CELLS,
    };
    use crate::shared::worldfield_testkit::{
        payload_has_solid_material, random_chunk_key_in_bounds, random_chunk_payload,
        DeterministicRng,
    };

    fn world_field_from_world(
        world: RegionChunkWorld,
        seed: u64,
        procgen: bool,
    ) -> ServerWorldField {
        let base_kind = world.base_kind();
        let was_dirty = world.any_dirty();
        let chunk_payloads: Vec<([i32; 4], ChunkPayload)> = world
            .chunks
            .iter()
            .map(|(&pos, chunk)| {
                (
                    [pos.x, pos.y, pos.z, pos.w],
                    ChunkPayload::from_chunk_compact(chunk),
                )
            })
            .collect();
        let mut field = ServerWorldField::from_chunk_payloads(
            base_kind,
            chunk_payloads,
            seed,
            procgen,
            HashSet::new(),
        );
        field.world_dirty = was_dirty;
        field
    }

    fn find_procgen_chunk(seed: u64) -> ChunkPos {
        let (min_y, max_y) = procgen::structure_chunk_y_bounds();
        for radius in 0..=24 {
            for x in -radius..=radius {
                for z in -radius..=radius {
                    for w in -radius..=radius {
                        for y in min_y..=max_y {
                            let chunk_pos = ChunkPos::new(x, y, z, w);
                            if procgen::structure_chunk_has_content_with_keepout(
                                seed, chunk_pos, None,
                            ) {
                                return chunk_pos;
                            }
                        }
                    }
                }
            }
        }
        panic!("failed to find a procgen structure chunk in search radius");
    }

    fn procgen_query_bounds(center: ChunkPos, radius: i32) -> Aabb4i {
        let r = radius.max(0);
        Aabb4i::new(
            [center.x - r, center.y - r, center.z - r, center.w - r],
            [center.x + r, center.y + r, center.z + r, center.w + r],
        )
    }

    #[test]
    fn apply_chunk_payloads_sets_chunks_without_marking_world_dirty() {
        let mut field = ServerWorldField::from_chunk_payloads(
            BaseWorldKind::Empty,
            Vec::<([i32; 4], ChunkPayload)>::new(),
            1337,
            false,
            HashSet::new(),
        );
        assert!(!field.any_dirty());

        let changed = field.apply_chunk_payloads(vec![([4, -2, 9, 0], ChunkPayload::Uniform(7))]);
        assert_eq!(changed, 1);
        assert!(field.has_chunk(ChunkPos::new(4, -2, 9, 0)));
        assert!(!field.any_dirty());
    }

    #[test]
    fn realize_chunk_preserves_explicit_empty_chunk() {
        let mut world = RegionChunkWorld::new_with_base(BaseWorldKind::FlatFloor {
            material: VoxelType(11),
        });
        let floor_chunk = ChunkPos::new(0, -1, 0, 0);
        world.insert_chunk(floor_chunk, Chunk::new());

        let mut field = world_field_from_world(world, 1337, false);
        let payload = field.realize_chunk(
            ChunkKey::from_chunk_pos(floor_chunk),
            RealizeProfile::Render,
        );

        assert!(matches!(payload, ChunkPayload::Dense16 { .. }));
        let dense = payload.dense_materials().expect("dense payload");
        assert!(dense.iter().all(|v| *v == 0));
    }

    #[test]
    fn realize_chunk_uses_flat_floor_base_when_no_chunk_exists() {
        let world = RegionChunkWorld::new_with_base(BaseWorldKind::FlatFloor {
            material: VoxelType(9),
        });
        let mut field = world_field_from_world(world, 1337, false);

        let payload = field.realize_chunk(
            ChunkKey::from_chunk_pos(ChunkPos::new(0, -1, 0, 0)),
            RealizeProfile::Render,
        );
        let dense = payload.dense_materials().expect("dense payload");
        assert!(dense.iter().any(|v| *v == 9));
    }

    #[test]
    fn query_content_bounds_matches_exact_non_empty_bounds_without_procgen() {
        let mut world = RegionChunkWorld::new();
        world.set_voxel(16, 0, -1, 9, VoxelType(4));
        world.set_voxel(-8, 10, 2, 2, VoxelType(5));

        let field = world_field_from_world(world, 1337, false);
        let query = QueryVolume {
            bounds: Aabb4i::new([-4, -4, -4, -4], [4, 4, 4, 4]),
        };

        let bounds = field.query_content_bounds(query).expect("content bounds");
        let (chunk_a, _) = world_to_chunk(16, 0, -1, 9);
        let (chunk_b, _) = world_to_chunk(-8, 10, 2, 2);
        let expected = Aabb4i::new(
            [
                chunk_a.x.min(chunk_b.x),
                chunk_a.y.min(chunk_b.y),
                chunk_a.z.min(chunk_b.z),
                chunk_a.w.min(chunk_b.w),
            ],
            [
                chunk_a.x.max(chunk_b.x),
                chunk_a.y.max(chunk_b.y),
                chunk_a.z.max(chunk_b.z),
                chunk_a.w.max(chunk_b.w),
            ],
        );
        assert_eq!(bounds, expected);
    }

    #[test]
    fn query_content_bounds_uses_conservative_fallback_for_large_queries() {
        let world = RegionChunkWorld::new_with_base(BaseWorldKind::FlatFloor {
            material: VoxelType(7),
        });
        let field = world_field_from_world(world, 1337, false);

        let extent = ((QUERY_CONTENT_BOUNDS_SCAN_BUDGET_CELLS as f64).powf(0.25) as i32) + 8;
        let query = QueryVolume {
            bounds: Aabb4i::new([-extent, -2, -extent, -extent], [extent, 2, extent, extent]),
        };

        let bounds = field
            .query_content_bounds(query)
            .expect("conservative bounds");
        assert_eq!(bounds, query.bounds);
    }

    #[test]
    fn procgen_chunk_realization_is_controlled_by_toggle() {
        let seed = 1337;
        let chunk_pos = find_procgen_chunk(seed);
        let mut disabled = world_field_from_world(RegionChunkWorld::new(), seed, false);
        let mut enabled = world_field_from_world(RegionChunkWorld::new(), seed, true);

        let disabled_payload =
            disabled.realize_chunk(ChunkKey::from_chunk_pos(chunk_pos), RealizeProfile::Render);
        let enabled_payload =
            enabled.realize_chunk(ChunkKey::from_chunk_pos(chunk_pos), RealizeProfile::Render);

        assert!(matches!(disabled_payload, ChunkPayload::Empty));
        let dense = enabled_payload
            .dense_materials()
            .expect("enabled payload should decode");
        assert!(dense.iter().any(|v| *v != 0));
    }

    #[test]
    fn query_region_core_returns_empty_node_when_no_content_exists() {
        let world = RegionChunkWorld::new();
        let field = world_field_from_world(world, 1337, false);

        let query = QueryVolume {
            bounds: Aabb4i::new([-1, -1, -1, -1], [1, 1, 1, 1]),
        };
        let tree = field.query_region_core(query, QueryDetail::Exact);
        assert!(matches!(tree.kind, RegionNodeKind::Empty));
    }

    #[test]
    fn query_region_core_materializes_sparse_branch_for_non_empty_queries() {
        let mut world = RegionChunkWorld::new();
        world.set_voxel(0, 0, 0, 0, VoxelType(6));

        let field = world_field_from_world(world, 1337, false);
        let query = QueryVolume {
            bounds: Aabb4i::new([0, 0, 0, 0], [1, 1, 1, 1]),
        };
        let tree = field.query_region_core(query, QueryDetail::Exact);

        match &tree.kind {
            RegionNodeKind::Branch(children) => {
                assert_eq!(children.len(), 1);
                let child = &children[0];
                assert_eq!(child.bounds, Aabb4i::new([0, 0, 0, 0], [0, 0, 0, 0]));
                let RegionNodeKind::ChunkArray(chunk_array) = &child.kind else {
                    panic!("expected child chunk array, got {:?}", child.kind);
                };
                assert_eq!(
                    chunk_array.index_codec,
                    ChunkArrayIndexCodec::PagedSparseRle
                );
                let dense_indices = chunk_array.decode_dense_indices().expect("decode indices");
                assert_eq!(dense_indices.len(), 1);
                assert!(dense_indices.iter().any(|idx| *idx != 0));
            }
            other => panic!("expected sparse branch node, got {other:?}"),
        }
    }

    #[test]
    fn query_region_core_flat_floor_emits_sparse_floor_slice() {
        let world = RegionChunkWorld::new_with_base(BaseWorldKind::FlatFloor {
            material: VoxelType(9),
        });
        let field = world_field_from_world(world, 1337, false);
        let query = QueryVolume {
            bounds: Aabb4i::new([-3, -2, -3, -3], [3, 2, 3, 3]),
        };
        let tree = field.query_region_core(query, QueryDetail::Exact);

        match &tree.kind {
            RegionNodeKind::Branch(children) => {
                assert_eq!(children.len(), 1);
                let floor_slice = &children[0];
                assert_eq!(floor_slice.bounds.min[1], -1);
                assert_eq!(floor_slice.bounds.max[1], -1);
                assert_eq!(floor_slice.bounds.min[0], -3);
                assert_eq!(floor_slice.bounds.max[0], 3);
                let RegionNodeKind::ChunkArray(chunk_array) = &floor_slice.kind else {
                    panic!(
                        "expected floor slice chunk array, got {:?}",
                        floor_slice.kind
                    );
                };
                assert_eq!(
                    chunk_array.index_codec,
                    ChunkArrayIndexCodec::PagedSparseRle
                );
                let dense_indices = chunk_array.decode_dense_indices().expect("decode indices");
                assert_eq!(dense_indices.len(), 7 * 7 * 7);
                assert!(dense_indices.iter().all(|idx| *idx == 0));
                assert_eq!(chunk_array.chunk_palette.len(), 1);
            }
            other => panic!("expected sparse floor branch, got {other:?}"),
        }
    }

    #[test]
    fn apply_voxel_edit_sets_and_prunes_chunk_against_virgin_world() {
        let world = RegionChunkWorld::new_with_base(BaseWorldKind::FlatFloor {
            material: VoxelType(11),
        });
        let mut field = world_field_from_world(world, 1337, false);

        let world_pos = [0, -1, 0, 0];
        let Some(chunk_pos) = field.apply_voxel_edit(world_pos, VoxelType::AIR) else {
            panic!("edit should change a floor voxel");
        };
        assert!(field.has_chunk(chunk_pos));

        let _ = field.apply_voxel_edit(world_pos, VoxelType(11));
        assert!(!field.has_chunk(chunk_pos));
    }

    #[test]
    fn query_region_core_matches_realized_non_empty_chunks_after_randomized_edits() {
        let mut field = world_field_from_world(RegionChunkWorld::new(), 2026, false);
        let domain = Aabb4i::new([-4, -2, -4, -4], [4, 2, 4, 4]);
        let mut rng = DeterministicRng::new(0x5157_4f52_4c44_544b);

        for _ in 0..220 {
            let key = random_chunk_key_in_bounds(&mut rng, domain);
            if rng.chance(1, 5) {
                let _ = field.remove_chunk(key.to_chunk_pos());
            } else {
                let payload = random_chunk_payload(&mut rng);
                let chunk = payload.to_voxel_chunk().expect("random payload decode");
                let _ = field.set_chunk(key.to_chunk_pos(), chunk);
            }
        }

        let core = field.query_region_core(QueryVolume { bounds: domain }, QueryDetail::Exact);
        let mut from_tree = crate::shared::worldfield::collect_non_empty_chunks_from_core_in_bounds(
            core.as_ref(),
            domain,
        );
        from_tree.sort_unstable_by_key(|(key, _)| key.pos);

        let mut expected = Vec::new();
        for w in domain.min[3]..=domain.max[3] {
            for z in domain.min[2]..=domain.max[2] {
                for y in domain.min[1]..=domain.max[1] {
                    for x in domain.min[0]..=domain.max[0] {
                        let key = ChunkKey { pos: [x, y, z, w] };
                        let payload = field.realize_chunk(key, RealizeProfile::Render);
                        if payload_has_solid_material(&payload) {
                            expected.push((key, payload));
                        }
                    }
                }
            }
        }
        expected.sort_unstable_by_key(|(key, _)| key.pos);

        assert_eq!(from_tree, expected);
    }

    #[test]
    fn procgen_query_region_core_is_stable_for_fixed_bounds() {
        let seed = 1337;
        let center = find_procgen_chunk(seed);
        let field = world_field_from_world(RegionChunkWorld::new(), seed, true);
        let bounds = procgen_query_bounds(center, 2);
        let query = QueryVolume { bounds };

        let first = field.query_region_core(query, QueryDetail::Exact);
        let first_chunks = collect_non_empty_chunks_from_core_in_bounds(first.as_ref(), bounds);
        assert!(
            !first_chunks.is_empty(),
            "expected non-empty procgen bounds around {:?}",
            center
        );

        for _ in 0..6 {
            let next = field.query_region_core(query, QueryDetail::Exact);
            let next_chunks = collect_non_empty_chunks_from_core_in_bounds(next.as_ref(), bounds);
            assert_eq!(next_chunks, first_chunks);
        }
    }

    #[test]
    fn procgen_working_set_refresh_is_idempotent_for_stationary_bounds() {
        let seed = 1337;
        let center = find_procgen_chunk(seed);
        let field = world_field_from_world(RegionChunkWorld::new(), seed, true);
        let bounds = procgen_query_bounds(center, 2);
        let query = QueryVolume { bounds };
        let mut working = RegionTreeWorkingSet::new();

        let first_core = field.query_region_core(query, QueryDetail::Exact);
        let first = working.refresh_from_core(bounds, first_core.as_ref());
        assert!(
            first.changed_bounds.is_some(),
            "expected first refresh to load procgen chunks"
        );

        for _ in 0..6 {
            let next_core = field.query_region_core(query, QueryDetail::Exact);
            let refresh = working.refresh_from_core(bounds, next_core.as_ref());
            assert!(
                refresh.changed_bounds.is_none(),
                "stationary refresh should not emit additional loads"
            );
        }
    }
}
