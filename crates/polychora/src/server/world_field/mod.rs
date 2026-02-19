use crate::shared::chunk_payload::ChunkPayload;
use crate::shared::region_tree::RegionTreeCore;
use crate::shared::spatial::Aabb4i;
use crate::shared::voxel::{BaseWorldKind, ChunkPos, VoxelType, CHUNK_VOLUME};
use std::collections::HashSet;
use std::sync::Arc;

mod legacy_generator;

pub use legacy_generator::LegacyWorldGenerator;

#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum QueryDetail {
    Coarse,
    Exact,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub struct QueryVolume {
    pub bounds: Aabb4i,
}

pub trait WorldField {
    fn query_region_core(&self, query: QueryVolume, detail: QueryDetail) -> Arc<RegionTreeCore>;
}

pub trait WorldOverlay: WorldField {}

#[derive(Debug)]
pub struct PassthroughWorldOverlay<F> {
    field: F,
}

impl<F> PassthroughWorldOverlay<F> {
    pub fn new(field: F) -> Self {
        Self { field }
    }

    pub fn field(&self) -> &F {
        &self.field
    }

    pub fn field_mut(&mut self) -> &mut F {
        &mut self.field
    }

    pub fn into_inner(self) -> F {
        self.field
    }
}

impl<F> WorldField for PassthroughWorldOverlay<F>
where
    F: WorldField,
{
    fn query_region_core(&self, query: QueryVolume, detail: QueryDetail) -> Arc<RegionTreeCore> {
        self.field.query_region_core(query, detail)
    }
}

impl<F> WorldOverlay for PassthroughWorldOverlay<F> where F: WorldField {}

impl PassthroughWorldOverlay<LegacyWorldGenerator> {
    pub fn from_chunk_payloads(
        base_kind: BaseWorldKind,
        chunk_payloads: impl IntoIterator<Item = ([i32; 4], ChunkPayload)>,
        world_seed: u64,
        procgen_structures: bool,
        blocked_cells: HashSet<crate::server::procgen::StructureCell>,
    ) -> Self {
        Self::new(LegacyWorldGenerator::from_chunk_payloads(
            base_kind,
            chunk_payloads,
            world_seed,
            procgen_structures,
            blocked_cells,
        ))
    }

    pub fn world_seed(&self) -> u64 {
        self.field.world_seed()
    }

    pub fn non_empty_chunk_count(&self) -> usize {
        self.field.non_empty_chunk_count()
    }

    pub fn clear_dirty(&mut self) {
        self.field.clear_dirty();
    }

    pub fn set_procgen_blocked_cells(
        &mut self,
        blocked_cells: HashSet<crate::server::procgen::StructureCell>,
    ) {
        self.field.set_procgen_blocked_cells(blocked_cells);
    }

    pub fn rebuild_procgen_keepout_from_chunks(&mut self, padding_chunks: i32) -> usize {
        self.field.rebuild_procgen_keepout_from_chunks(padding_chunks)
    }

    pub fn prune_virgin_chunks(&mut self) -> usize {
        self.field.prune_virgin_chunks()
    }

    pub fn apply_voxel_edit(
        &mut self,
        position: [i32; 4],
        material: VoxelType,
    ) -> Option<ChunkPos> {
        self.field.apply_voxel_edit(position, material)
    }

    pub fn chunk_at(&self, chunk_pos: ChunkPos) -> Option<[VoxelType; CHUNK_VOLUME]> {
        self.field.chunk_at(chunk_pos)
    }

    pub fn effective_chunk(
        &self,
        chunk_pos: ChunkPos,
        preserve_explicit_empty_chunk: bool,
    ) -> Option<[VoxelType; CHUNK_VOLUME]> {
        self.field
            .effective_chunk(chunk_pos, preserve_explicit_empty_chunk)
    }
}

pub type ServerWorldOverlay = PassthroughWorldOverlay<LegacyWorldGenerator>;
