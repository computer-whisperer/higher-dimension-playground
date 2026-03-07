use super::{QueryDetail, QueryVolume, WorldField};
use crate::server::procgen;
use crate::server::procgen_wasm::{xzw_orientation_to_tesseract, ProcgenWasmState};
use crate::shared::chunk_payload::ChunkPayload;
use crate::shared::region_tree::{RegionChunkTree, RegionNodeKind, RegionTreeCore};
use crate::shared::spatial::{Aabb4i, ChunkCoord};
use crate::shared::voxel::{BaseWorldKind, BlockData, CHUNK_SIZE};
use std::cell::RefCell;
use std::collections::HashSet;
use std::sync::Arc;

const FLAT_FLOOR_CHUNK_Y: i32 = -1;

pub struct FlatWorldGenerator {
    flat_floor_voxel: Option<BlockData>,
    world_seed: u64,
    procgen_structures: bool,
    blocked_cells: HashSet<procgen::StructureCell>,
    procgen_wasm: Option<RefCell<ProcgenWasmState>>,
}

impl std::fmt::Debug for FlatWorldGenerator {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("FlatWorldGenerator")
            .field("flat_floor_voxel", &self.flat_floor_voxel)
            .field("world_seed", &self.world_seed)
            .field("procgen_structures", &self.procgen_structures)
            .field("blocked_cells", &self.blocked_cells)
            .field("procgen_wasm", &self.procgen_wasm.as_ref().map(|_| ".."))
            .finish()
    }
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
            procgen_wasm: None,
        }
    }

    pub fn set_procgen_wasm(&mut self, state: ProcgenWasmState) {
        self.procgen_wasm = Some(RefCell::new(state));
    }

    fn procgen_keepout_cells(&self) -> Option<&HashSet<procgen::StructureCell>> {
        if self.blocked_cells.is_empty() {
            None
        } else {
            Some(&self.blocked_cells)
        }
    }

    fn floor_core_for_bounds(&self, bounds: Aabb4i) -> Option<RegionTreeCore> {
        let material = self.flat_floor_voxel.as_ref()?;
        let cs = ChunkCoord::from_num(CHUNK_SIZE as i32);
        let floor_y_world_min = ChunkCoord::from_num(FLAT_FLOOR_CHUNK_Y).saturating_mul(cs);
        let floor_y_world_max = floor_y_world_min.saturating_add(cs);
        if !bounds.is_valid()
            || floor_y_world_min >= bounds.max[1]
            || floor_y_world_max <= bounds.min[1]
        {
            return None;
        }
        let floor_bounds = Aabb4i::new(
            [bounds.min[0], floor_y_world_min, bounds.min[2], bounds.min[3]],
            [bounds.max[0], floor_y_world_max, bounds.max[2], bounds.max[3]],
        );
        Some(RegionTreeCore {
            bounds: floor_bounds,
            kind: RegionNodeKind::Uniform(material.clone()),
            generator_version_hash: 0,
        })
    }

    fn insert_procgen_chunks_for_bounds(&self, bounds: Aabb4i, tree: &mut RegionChunkTree) {
        if !self.procgen_structures || !bounds.is_valid() {
            return;
        }
        let wasm_cell = match &self.procgen_wasm {
            Some(cell) => cell,
            None => return,
        };
        let mut wasm = wasm_cell.borrow_mut();
        let config = wasm.structure_placement_config();

        let reports = procgen::collect_structure_placement_reports(
            self.world_seed,
            bounds,
            self.procgen_keepout_cells(),
            None,
            None,
            &config,
        );
        for report in reports {
            let decls = wasm.caller.declarations();
            if report.blueprint_idx >= decls.len() {
                continue;
            }
            let structure_id = decls[report.blueprint_idx].id;
            let orientation = xzw_orientation_to_tesseract(report.orientation);
            match wasm.prepare_and_generate(structure_id, report.cell_hash, orientation, report.origin) {
                Ok(core) if core.bounds.is_valid() => {
                    let _ = tree.splice_non_empty_core_in_bounds(core.bounds, &core);
                }
                Err(e) => {
                    eprintln!("procgen WASM error: {e}");
                }
                _ => {}
            }
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

#[cfg(test)]
mod tests {
    use super::*;
    use crate::shared::region_tree::collect_non_empty_chunks_from_core_in_bounds;
    use crate::shared::voxel::BlockData;

    fn grid_floor_block() -> BlockData {
        use polychora_plugin_api::content_ids;
        BlockData::simple(content_ids::CONTENT_NS, content_ids::BLOCK_GRID_FLOOR)
    }

    #[test]
    fn flat_floor_single_chunk_query_returns_uniform_payload() {
        let floor = grid_floor_block();
        let generator = FlatWorldGenerator::from_chunk_payloads(
            BaseWorldKind::FlatFloor {
                material: floor.clone(),
            },
            Vec::<([i32; 4], ChunkPayload)>::new(),
            123,
            false,
            HashSet::new(),
        );
        let cs = CHUNK_SIZE as i32;
        let bounds = Aabb4i::from_i32(
            [0, FLAT_FLOOR_CHUNK_Y * cs, 0, 0],
            [cs, (FLAT_FLOOR_CHUNK_Y + 1) * cs, cs, cs],
        );
        let core = generator.query_region_core(QueryVolume { bounds }, QueryDetail::Exact);
        assert_eq!(core.bounds, bounds);
        assert_eq!(core.kind, RegionNodeKind::Uniform(floor));
    }

    #[test]
    fn flat_floor_multi_chunk_query_emits_only_floor_slice() {
        let floor = grid_floor_block();
        let generator = FlatWorldGenerator::from_chunk_payloads(
            BaseWorldKind::FlatFloor {
                material: floor.clone(),
            },
            Vec::<([i32; 4], ChunkPayload)>::new(),
            123,
            false,
            HashSet::new(),
        );
        let cs = CHUNK_SIZE as i32;
        let bounds = Aabb4i::from_i32(
            [-2 * cs, -3 * cs, -1 * cs, -2 * cs],
            [3 * cs, 2 * cs, 4 * cs, 3 * cs],
        );
        let core = generator.query_region_core(QueryVolume { bounds }, QueryDetail::Exact);
        let non_empty = collect_non_empty_chunks_from_core_in_bounds(core.as_ref(), bounds);
        let expected_count = 5usize * 5 * 5;
        assert_eq!(non_empty.len(), expected_count);
        assert!(non_empty.iter().all(|(key, payload)| key[1]
            == ChunkCoord::from_num(FLAT_FLOOR_CHUNK_Y)
            && payload.uniform_block() == Some(&floor)));
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
        let cs = CHUNK_SIZE as i32;
        let bounds = Aabb4i::from_i32(
            [-2 * cs, -3 * cs, -1 * cs, -2 * cs],
            [3 * cs, 2 * cs, 4 * cs, 3 * cs],
        );
        let core = generator.query_region_core(QueryVolume { bounds }, QueryDetail::Exact);
        assert_eq!(core.bounds, bounds);
        assert!(matches!(core.kind, RegionNodeKind::Empty));
    }
}
