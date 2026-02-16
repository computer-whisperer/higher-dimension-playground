use super::procgen;
use crate::shared::voxel::{BaseWorldKind, Chunk, ChunkPos, VoxelType, CHUNK_SIZE};
use crate::shared::worldfield::{
    Aabb4i, ChunkArrayData, ChunkKey, ChunkPayload, QueryDetail, QueryVolume, RealizeProfile,
    RegionNodeKind, RegionOverrideTree, RegionTreeCore, WorldField,
    QUERY_CONTENT_BOUNDS_SCAN_BUDGET_CELLS,
};
use std::collections::{HashMap, HashSet};
use std::sync::Arc;

#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
struct RealizeCacheKey {
    chunk_key: ChunkKey,
    profile: RealizeProfile,
}

#[derive(Debug)]
pub struct LegacyWorldField<'a> {
    base_kind: BaseWorldKind,
    flat_floor_chunk: Chunk,
    override_tree: Option<&'a RegionOverrideTree>,
    world_seed: u64,
    procgen_structures: bool,
    blocked_cells: Option<&'a HashSet<procgen::StructureCell>>,
    realize_cache: HashMap<RealizeCacheKey, ChunkPayload>,
}

impl<'a> LegacyWorldField<'a> {
    pub fn new(
        base_kind: BaseWorldKind,
        override_tree: Option<&'a RegionOverrideTree>,
        world_seed: u64,
        procgen_structures: bool,
        blocked_cells: Option<&'a HashSet<procgen::StructureCell>>,
    ) -> Self {
        let flat_floor_chunk = match base_kind {
            BaseWorldKind::FlatFloor { material } => build_flat_floor_chunk(material),
            BaseWorldKind::Empty => Chunk::new(),
        };
        Self {
            base_kind,
            flat_floor_chunk,
            override_tree,
            world_seed,
            procgen_structures,
            blocked_cells,
            realize_cache: HashMap::new(),
        }
    }

    pub fn clear_realize_cache(&mut self) {
        self.realize_cache.clear();
    }

    fn clone_base_chunk_or_empty_at(&self, chunk_pos: ChunkPos) -> Chunk {
        match self.base_kind {
            BaseWorldKind::Empty => Chunk::new(),
            BaseWorldKind::FlatFloor { .. } if chunk_pos.y == -1 => self.flat_floor_chunk.clone(),
            BaseWorldKind::FlatFloor { .. } => Chunk::new(),
        }
    }

    fn base_chunk_y_bounds_for_scale(&self, chunk_scale: i32) -> Option<(i32, i32)> {
        let scale = chunk_scale.max(1);
        match self.base_kind {
            BaseWorldKind::Empty => None,
            BaseWorldKind::FlatFloor { .. } => {
                if self.flat_floor_chunk.is_empty() {
                    None
                } else {
                    let y = (-1i32).div_euclid(scale);
                    Some((y, y))
                }
            }
        }
    }

    fn override_chunk_at(&self, chunk_pos: ChunkPos) -> Option<Chunk> {
        self.override_tree.and_then(|tree| {
            tree.chunk_override_payload(ChunkKey::from_chunk_pos(chunk_pos))
                .and_then(|payload| payload.to_voxel_chunk().ok())
        })
    }

    fn build_effective_chunk(
        &self,
        chunk_pos: ChunkPos,
        preserve_explicit_empty_override: bool,
    ) -> Option<Chunk> {
        if let Some(override_chunk) = self.override_chunk_at(chunk_pos) {
            if preserve_explicit_empty_override {
                return Some(override_chunk);
            }
            return (!override_chunk.is_empty()).then_some(override_chunk);
        }

        let mut chunk = self.clone_base_chunk_or_empty_at(chunk_pos);
        if self.procgen_structures {
            if let Some(structure_chunk) = procgen::generate_structure_chunk_with_keepout(
                self.world_seed,
                chunk_pos,
                self.blocked_cells,
            ) {
                merge_non_air_voxels(&mut chunk, &structure_chunk);
            }
        }
        (!chunk.is_empty()).then_some(chunk)
    }

    fn might_have_content_conservatively(&self, bounds: Aabb4i) -> bool {
        if !bounds.is_valid() {
            return false;
        }

        // Any non-empty override inside bounds is definite content.
        if self
            .override_tree
            .map(|tree| tree.any_non_empty_override_in_bounds(bounds))
            .unwrap_or(false)
        {
            return true;
        }

        // Base content is infinite in x/z/w and bounded in y.
        if let Some((base_min_y, base_max_y)) = self.base_chunk_y_bounds_for_scale(1) {
            if bounds.max[1] >= base_min_y && bounds.min[1] <= base_max_y {
                return true;
            }
        }

        // Procgen content can appear anywhere in x/z/w, bounded in y.
        if self.procgen_structures {
            let (procgen_min_y, procgen_max_y) = procgen::structure_chunk_y_bounds();
            if bounds.max[1] >= procgen_min_y && bounds.min[1] <= procgen_max_y {
                return true;
            }
        }

        false
    }
}

impl WorldField for LegacyWorldField<'_> {
    fn query_region_core(&self, query: QueryVolume, _detail: QueryDetail) -> Arc<RegionTreeCore> {
        let bounds = query.bounds;
        if !bounds.is_valid() {
            return Arc::new(RegionTreeCore {
                bounds,
                kind: RegionNodeKind::Empty,
                generator_version_hash: 0,
            });
        }

        let Some(cell_count) = bounds.chunk_cell_count() else {
            return Arc::new(RegionTreeCore {
                bounds,
                kind: RegionNodeKind::Empty,
                generator_version_hash: 0,
            });
        };

        let mut palette = vec![ChunkPayload::Empty];
        let mut palette_lookup = HashMap::<ChunkPayload, u16>::new();
        palette_lookup.insert(ChunkPayload::Empty, 0);

        let mut dense_indices = Vec::with_capacity(cell_count);

        for w in bounds.min[3]..=bounds.max[3] {
            for z in bounds.min[2]..=bounds.max[2] {
                for y in bounds.min[1]..=bounds.max[1] {
                    for x in bounds.min[0]..=bounds.max[0] {
                        let chunk_pos = ChunkPos::new(x, y, z, w);
                        let palette_idx =
                            if let Some(chunk) = self.build_effective_chunk(chunk_pos, false) {
                                let payload = ChunkPayload::from_chunk_dense(&chunk);
                                if let Some(existing) = palette_lookup.get(&payload) {
                                    *existing
                                } else {
                                    let next = palette.len() as u16;
                                    palette_lookup.insert(payload.clone(), next);
                                    palette.push(payload);
                                    next
                                }
                            } else {
                                0
                            };
                        dense_indices.push(palette_idx);
                    }
                }
            }
        }

        if dense_indices.iter().all(|idx| *idx == 0) {
            return Arc::new(RegionTreeCore {
                bounds,
                kind: RegionNodeKind::Empty,
                generator_version_hash: 0,
            });
        }

        let chunk_array =
            ChunkArrayData::from_dense_indices(bounds, palette, dense_indices, Some(0))
                .expect("query_region_core: chunk array encoding should be valid");

        Arc::new(RegionTreeCore {
            bounds,
            kind: RegionNodeKind::ChunkArray(chunk_array),
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
                        if self.build_effective_chunk(chunk_pos, false).is_none() {
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
        let payload = if let Some(chunk) = self.build_effective_chunk(chunk_pos, true) {
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

#[cfg(test)]
mod tests {
    use super::*;
    use crate::shared::voxel::{world_to_chunk, BaseWorldKind, Chunk, VoxelType, VoxelWorld};
    use crate::shared::worldfield::{
        ChunkArrayIndexCodec, RegionOverrideTree, QUERY_CONTENT_BOUNDS_SCAN_BUDGET_CELLS,
    };

    fn override_tree_from_world(world: &VoxelWorld) -> RegionOverrideTree {
        RegionOverrideTree::from_chunk_overrides(world.chunks.iter().map(|(&pos, chunk)| {
            (
                ChunkKey::from_chunk_pos(pos),
                ChunkPayload::from_chunk_compact(chunk),
            )
        }))
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

    #[test]
    fn realize_chunk_preserves_explicit_empty_override() {
        let mut world = VoxelWorld::new_with_base(BaseWorldKind::FlatFloor {
            material: VoxelType(11),
        });
        let floor_chunk = ChunkPos::new(0, -1, 0, 0);
        world.insert_chunk(floor_chunk, Chunk::new());

        let overrides = override_tree_from_world(&world);
        let mut field =
            LegacyWorldField::new(world.base_kind(), Some(&overrides), 1337, false, None);
        let payload = field.realize_chunk(
            ChunkKey::from_chunk_pos(floor_chunk),
            RealizeProfile::Render,
        );

        assert!(matches!(payload, ChunkPayload::Dense16 { .. }));
        let dense = payload.dense_materials().expect("dense payload");
        assert!(dense.iter().all(|v| *v == 0));
    }

    #[test]
    fn realize_chunk_uses_flat_floor_base_when_no_override_exists() {
        let world = VoxelWorld::new_with_base(BaseWorldKind::FlatFloor {
            material: VoxelType(9),
        });
        let mut field = LegacyWorldField::new(world.base_kind(), None, 1337, false, None);

        let payload = field.realize_chunk(
            ChunkKey::from_chunk_pos(ChunkPos::new(0, -1, 0, 0)),
            RealizeProfile::Render,
        );
        let dense = payload.dense_materials().expect("dense payload");
        assert!(dense.iter().any(|v| *v == 9));
    }

    #[test]
    fn query_content_bounds_matches_exact_non_empty_bounds_without_procgen() {
        let mut world = VoxelWorld::new();
        world.set_voxel(16, 0, -1, 9, VoxelType(4));
        world.set_voxel(-8, 10, 2, 2, VoxelType(5));

        let overrides = override_tree_from_world(&world);
        let field = LegacyWorldField::new(world.base_kind(), Some(&overrides), 1337, false, None);
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
        let world = VoxelWorld::new_with_base(BaseWorldKind::FlatFloor {
            material: VoxelType(7),
        });
        let field = LegacyWorldField::new(world.base_kind(), None, 1337, false, None);

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
        let world = VoxelWorld::new();

        let mut disabled = LegacyWorldField::new(world.base_kind(), None, seed, false, None);
        let mut enabled = LegacyWorldField::new(world.base_kind(), None, seed, true, None);

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
        let world = VoxelWorld::new();
        let field = LegacyWorldField::new(world.base_kind(), None, 1337, false, None);

        let query = QueryVolume {
            bounds: Aabb4i::new([-1, -1, -1, -1], [1, 1, 1, 1]),
        };
        let tree = field.query_region_core(query, QueryDetail::Exact);
        assert!(matches!(tree.kind, RegionNodeKind::Empty));
    }

    #[test]
    fn query_region_core_materializes_chunk_array_for_non_empty_queries() {
        let mut world = VoxelWorld::new();
        world.set_voxel(0, 0, 0, 0, VoxelType(6));

        let overrides = override_tree_from_world(&world);
        let field = LegacyWorldField::new(world.base_kind(), Some(&overrides), 1337, false, None);
        let query = QueryVolume {
            bounds: Aabb4i::new([0, 0, 0, 0], [1, 1, 1, 1]),
        };
        let tree = field.query_region_core(query, QueryDetail::Exact);

        match &tree.kind {
            RegionNodeKind::ChunkArray(chunk_array) => {
                assert_eq!(
                    chunk_array.index_codec,
                    ChunkArrayIndexCodec::PagedSparseRle
                );
                let dense_indices = chunk_array.decode_dense_indices().expect("decode indices");
                assert_eq!(dense_indices.len(), 16);
                assert!(dense_indices.iter().any(|idx| *idx != 0));
                assert_eq!(chunk_array.bounds, query.bounds);
            }
            other => panic!("expected chunk array node, got {other:?}"),
        }
    }
}
