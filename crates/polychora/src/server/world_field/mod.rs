use crate::shared::chunk_payload::ChunkPayload;
use crate::shared::region_tree::{
    chunk_key_from_chunk_pos, chunk_pos_from_chunk_key, collect_non_empty_chunks_from_core_in_bounds,
    RegionChunkTree, RegionTreeCore,
};
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
    dirty_chunks: RegionChunkTree,
}

impl<F> PassthroughWorldOverlay<F> {
    pub fn new(field: F) -> Self {
        Self {
            field,
            dirty_chunks: RegionChunkTree::new(),
        }
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

    pub fn mark_dirty_chunk(&mut self, chunk_pos: ChunkPos) {
        let _ = self
            .dirty_chunks
            .set_chunk(chunk_key_from_chunk_pos(chunk_pos), Some(ChunkPayload::Uniform(1)));
    }

    pub fn clear_dirty_chunks(&mut self) {
        self.dirty_chunks = RegionChunkTree::new();
    }

    pub fn take_dirty_chunk_positions(&mut self) -> Vec<ChunkPos> {
        let Some(bounds) = self.dirty_chunks.root().map(|root| root.bounds) else {
            return Vec::new();
        };
        let dirty_core = self.dirty_chunks.take_non_empty_core_in_bounds(bounds);
        collect_non_empty_chunks_from_core_in_bounds(&dirty_core, dirty_core.bounds)
            .into_iter()
            .map(|(key, _)| chunk_pos_from_chunk_key(key))
            .collect()
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
        self.clear_dirty_chunks();
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
        let changed = self.field.apply_voxel_edit(position, material);
        if let Some(chunk_pos) = changed {
            self.mark_dirty_chunk(chunk_pos);
        }
        changed
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

#[cfg(test)]
mod tests {
    use super::*;
    use crate::shared::voxel::BaseWorldKind;

    #[test]
    fn overlay_dirty_chunk_drain_returns_touched_chunk_once() {
        let mut overlay = ServerWorldOverlay::from_chunk_payloads(
            BaseWorldKind::Empty,
            Vec::<([i32; 4], ChunkPayload)>::new(),
            123,
            false,
            HashSet::new(),
        );

        let changed_a = overlay.apply_voxel_edit([0, 0, 0, 0], VoxelType(3));
        let changed_b = overlay.apply_voxel_edit([1, 0, 0, 0], VoxelType(4));
        assert_eq!(changed_a, Some(ChunkPos::new(0, 0, 0, 0)));
        assert_eq!(changed_b, Some(ChunkPos::new(0, 0, 0, 0)));

        let dirty = overlay.take_dirty_chunk_positions();
        assert_eq!(dirty, vec![ChunkPos::new(0, 0, 0, 0)]);
        assert!(overlay.take_dirty_chunk_positions().is_empty());
    }

    #[test]
    fn overlay_dirty_chunk_drain_tracks_multiple_chunks_sorted() {
        let mut overlay = ServerWorldOverlay::from_chunk_payloads(
            BaseWorldKind::Empty,
            Vec::<([i32; 4], ChunkPayload)>::new(),
            123,
            false,
            HashSet::new(),
        );

        // Write into three separate chunks.
        let _ = overlay.apply_voxel_edit([0, 0, 0, 0], VoxelType(1));
        let _ = overlay.apply_voxel_edit([8, 0, 0, 0], VoxelType(1));
        let _ = overlay.apply_voxel_edit([0, 8, 0, 0], VoxelType(1));

        let dirty = overlay.take_dirty_chunk_positions();
        assert_eq!(
            dirty,
            vec![
                ChunkPos::new(0, 0, 0, 0),
                ChunkPos::new(0, 1, 0, 0),
                ChunkPos::new(1, 0, 0, 0),
            ]
        );
    }

    #[test]
    fn overlay_clear_dirty_clears_overlay_dirty_chunks() {
        let mut overlay = ServerWorldOverlay::from_chunk_payloads(
            BaseWorldKind::Empty,
            Vec::<([i32; 4], ChunkPayload)>::new(),
            123,
            false,
            HashSet::new(),
        );

        let _ = overlay.apply_voxel_edit([0, 0, 0, 0], VoxelType(5));
        overlay.clear_dirty();
        assert!(overlay.take_dirty_chunk_positions().is_empty());
    }
}
