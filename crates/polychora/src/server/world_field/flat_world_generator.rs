use super::{QueryDetail, QueryVolume, WorldField};
use crate::materials::block_to_material_token;
use crate::server::procgen;
use crate::shared::chunk_payload::{ChunkPayload, ResolvedChunkPayload};
use crate::shared::region_tree::{
    chunk_key_from_chunk_pos, RegionChunkTree, RegionNodeKind, RegionTreeCore,
};
use crate::shared::spatial::Aabb4i;
use crate::shared::voxel::{BaseWorldKind, BlockData, ChunkPos, CHUNK_VOLUME};
use std::collections::HashSet;
use std::sync::Arc;

const FLAT_FLOOR_CHUNK_Y: i32 = -1;

#[derive(Debug)]
pub struct FlatWorldGenerator {
    flat_floor_voxel: Option<BlockData>,
    world_seed: u64,
    procgen_structures: bool,
    blocked_cells: HashSet<procgen::StructureCell>,
}

impl FlatWorldGenerator {
    pub fn from_chunk_payloads(
        base_kind: BaseWorldKind,
        _chunk_payloads: impl IntoIterator<Item = ([i32; 4], ChunkPayload)>,
        world_seed: u64,
        procgen_structures: bool,
        blocked_cells: HashSet<procgen::StructureCell>,
    ) -> Self {
        Self {
            flat_floor_voxel: floor_voxel_from_base_kind(base_kind),
            world_seed,
            procgen_structures,
            blocked_cells,
        }
    }

    fn procgen_keepout_cells(&self) -> Option<&HashSet<procgen::StructureCell>> {
        if self.blocked_cells.is_empty() {
            None
        } else {
            Some(&self.blocked_cells)
        }
    }

    fn base_payload_for_chunk_y(&self, chunk_y: i32) -> Option<ChunkPayload> {
        if chunk_y != FLAT_FLOOR_CHUNK_Y {
            return None;
        }
        self.flat_floor_voxel.as_ref().map(|material| {
            ChunkPayload::Uniform(block_to_material_token(material.namespace, material.block_type))
        })
    }

    fn base_dense_chunk_for_chunk_y(&self, chunk_y: i32) -> [u16; CHUNK_VOLUME] {
        if chunk_y != FLAT_FLOOR_CHUNK_Y {
            return [0u16; CHUNK_VOLUME];
        }
        self.flat_floor_voxel
            .as_ref()
            .map(|material| {
                let mat = block_to_material_token(material.namespace, material.block_type);
                [mat; CHUNK_VOLUME]
            })
            .unwrap_or([0u16; CHUNK_VOLUME])
    }

    fn floor_core_for_bounds(&self, bounds: Aabb4i) -> Option<RegionTreeCore> {
        let material = self.flat_floor_voxel.as_ref()?;
        if !bounds.is_valid()
            || FLAT_FLOOR_CHUNK_Y < bounds.min[1]
            || FLAT_FLOOR_CHUNK_Y > bounds.max[1]
        {
            return None;
        }
        let floor_bounds = Aabb4i::new(
            [
                bounds.min[0],
                FLAT_FLOOR_CHUNK_Y,
                bounds.min[2],
                bounds.min[3],
            ],
            [
                bounds.max[0],
                FLAT_FLOOR_CHUNK_Y,
                bounds.max[2],
                bounds.max[3],
            ],
        );
        Some(RegionTreeCore {
            bounds: floor_bounds,
            kind: RegionNodeKind::Uniform(material.clone()),
            generator_version_hash: 0,
        })
    }

    fn structure_chunk_payload(&self, chunk_pos: ChunkPos) -> Option<ChunkPayload> {
        if !self.procgen_structures {
            return None;
        }
        let structure_chunk = procgen::generate_structure_chunk_with_keepout(
            self.world_seed,
            chunk_pos,
            self.procgen_keepout_cells(),
        )?;
        let mut merged = self.base_dense_chunk_for_chunk_y(chunk_pos.y);
        merge_non_air_u16(&mut merged, &structure_chunk);
        let payload = payload_from_u16_chunk_compact(&merged);
        let base_payload = self.base_payload_for_chunk_y(chunk_pos.y);
        if base_payload
            .as_ref()
            .map(|base| canonicalize_payload(base.clone()) == canonicalize_payload(payload.clone()))
            .unwrap_or(false)
        {
            None
        } else {
            Some(payload)
        }
    }

    fn insert_procgen_chunks_for_bounds(&self, bounds: Aabb4i, tree: &mut RegionChunkTree) {
        if !self.procgen_structures || !bounds.is_valid() {
            return;
        }
        let candidates = procgen::structure_chunk_positions_for_bounds_with_keepout(
            self.world_seed,
            bounds,
            self.procgen_keepout_cells(),
        );
        for chunk_pos in candidates {
            let Some(payload) = self.structure_chunk_payload(chunk_pos) else {
                continue;
            };
            let resolved = ResolvedChunkPayload::from_legacy_payload(payload);
            let _ = tree.set_chunk(chunk_key_from_chunk_pos(chunk_pos), Some(resolved));
        }
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

        let mut tree = RegionChunkTree::new();
        if let Some(floor_core) = self.floor_core_for_bounds(bounds) {
            let _ = tree.splice_non_empty_core_in_bounds(floor_core.bounds, &floor_core);
        }
        self.insert_procgen_chunks_for_bounds(bounds, &mut tree);
        Arc::new(tree.root().cloned().unwrap_or(RegionTreeCore {
            bounds,
            kind: RegionNodeKind::Empty,
            generator_version_hash: 0,
        }))
    }
}

impl WorldField for FlatWorldGenerator {
    fn query_region_core(&self, query: QueryVolume, detail: QueryDetail) -> Arc<RegionTreeCore> {
        FlatWorldGenerator::query_region_core(self, query, detail)
    }
}

fn floor_voxel_from_base_kind(base_kind: BaseWorldKind) -> Option<BlockData> {
    match base_kind {
        BaseWorldKind::FlatFloor { material } if !material.is_air() => Some(material),
        BaseWorldKind::FlatFloor { .. }
        | BaseWorldKind::MassivePlatforms { .. }
        | BaseWorldKind::Empty => None,
    }
}

fn payload_from_u16_chunk_compact(chunk: &[u16; CHUNK_VOLUME]) -> ChunkPayload {
    let materials: Vec<u16> = chunk.to_vec();
    ChunkPayload::from_dense_materials_compact(&materials)
        .unwrap_or(ChunkPayload::Dense16 { materials })
}

fn canonicalize_payload(payload: ChunkPayload) -> ChunkPayload {
    let payload = match payload {
        ChunkPayload::Empty => ChunkPayload::Uniform(0),
        other => other,
    };
    let Ok(dense) = payload.dense_materials() else {
        return payload;
    };
    if dense.is_empty() {
        return payload;
    }
    let first = dense[0];
    if dense.iter().all(|material| *material == first) {
        ChunkPayload::Uniform(first)
    } else {
        payload
    }
}

fn merge_non_air_u16(dst: &mut [u16; CHUNK_VOLUME], src: &[u16; CHUNK_VOLUME]) {
    for (idx, &material) in src.iter().enumerate() {
        if material != 0 {
            dst[idx] = material;
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::shared::region_tree::collect_non_empty_chunks_from_core_in_bounds;
    use crate::shared::voxel::BlockData;

    #[test]
    fn flat_floor_single_chunk_query_returns_uniform_payload() {
        let generator = FlatWorldGenerator::from_chunk_payloads(
            BaseWorldKind::FlatFloor {
                material: BlockData::simple(0, 11),
            },
            Vec::<([i32; 4], ChunkPayload)>::new(),
            123,
            false,
            HashSet::new(),
        );
        let bounds = Aabb4i::new([0, FLAT_FLOOR_CHUNK_Y, 0, 0], [0, FLAT_FLOOR_CHUNK_Y, 0, 0]);
        let core = generator.query_region_core(QueryVolume { bounds }, QueryDetail::Exact);
        assert_eq!(core.bounds, bounds);
        assert_eq!(core.kind, RegionNodeKind::Uniform(crate::shared::voxel::BlockData::simple(0, 11)));
    }

    #[test]
    fn flat_floor_multi_chunk_query_emits_only_floor_slice() {
        let generator = FlatWorldGenerator::from_chunk_payloads(
            BaseWorldKind::FlatFloor {
                material: BlockData::simple(0, 11),
            },
            Vec::<([i32; 4], ChunkPayload)>::new(),
            123,
            false,
            HashSet::new(),
        );
        let bounds = Aabb4i::new([-2, -3, -1, -2], [2, 1, 3, 2]);
        let core = generator.query_region_core(QueryVolume { bounds }, QueryDetail::Exact);
        let non_empty = collect_non_empty_chunks_from_core_in_bounds(core.as_ref(), bounds);
        let expected_count = ((bounds.max[0] - bounds.min[0] + 1) as usize)
            * ((bounds.max[2] - bounds.min[2] + 1) as usize)
            * ((bounds.max[3] - bounds.min[3] + 1) as usize);
        assert_eq!(non_empty.len(), expected_count);
        assert!(non_empty
            .iter()
            .all(|(key, payload)| key[1] == FLAT_FLOOR_CHUNK_Y
                && payload.uniform_block() == Some(&BlockData::simple(0, 11))));
    }

    #[test]
    fn empty_world_query_returns_empty() {
        let generator = FlatWorldGenerator::from_chunk_payloads(
            BaseWorldKind::Empty,
            Vec::<([i32; 4], ChunkPayload)>::new(),
            123,
            false,
            HashSet::new(),
        );
        let bounds = Aabb4i::new([-2, -3, -1, -2], [2, 1, 3, 2]);
        let core = generator.query_region_core(QueryVolume { bounds }, QueryDetail::Exact);
        assert_eq!(core.bounds, bounds);
        assert!(matches!(core.kind, RegionNodeKind::Empty));
    }
}
