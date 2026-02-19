use crate::save_v4::{self, SaveChunkPayloadPatchRequest};
use crate::shared::chunk_payload::ChunkPayload;
use crate::shared::region_tree::{
    chunk_key_from_chunk_pos, RegionChunkTree, RegionNodeKind, RegionTreeCore,
};
use crate::shared::spatial::Aabb4i;
use crate::shared::voxel::{world_to_chunk, BaseWorldKind, ChunkPos, VoxelType, CHUNK_VOLUME};
use std::collections::HashSet;
use std::io;
use std::path::{Path, PathBuf};
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
struct SaveStreamingState {
    root: PathBuf,
    index: save_v4::IndexPayload,
    loaded_bounds: Vec<Aabb4i>,
    next_entity_id: u64,
}

#[derive(Debug)]
pub struct PassthroughWorldOverlay<F> {
    field: F,
    dirty_chunks: RegionChunkTree,
    dirty_save_chunks: HashSet<[i32; 4]>,
    save_stream: Option<SaveStreamingState>,
}

impl<F> PassthroughWorldOverlay<F> {
    pub fn new(field: F) -> Self {
        Self {
            field,
            dirty_chunks: RegionChunkTree::new(),
            dirty_save_chunks: HashSet::new(),
            save_stream: None,
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
        let _ = self.dirty_chunks.set_chunk(
            chunk_key_from_chunk_pos(chunk_pos),
            Some(ChunkPayload::Uniform(1)),
        );
    }

    pub fn clear_dirty_chunks(&mut self) {
        self.dirty_chunks = RegionChunkTree::new();
    }

    pub fn take_dirty_bounds(&mut self) -> Vec<Aabb4i> {
        let Some(bounds) = self.dirty_chunks.root().map(|root| root.bounds) else {
            return Vec::new();
        };
        let dirty_core = self.dirty_chunks.take_non_empty_core_in_bounds(bounds);
        let mut out = Vec::new();
        collect_non_empty_node_bounds(&dirty_core, &mut out);
        out.sort_unstable_by_key(|bounds| {
            (
                bounds.min[0],
                bounds.min[1],
                bounds.min[2],
                bounds.min[3],
                bounds.max[0],
                bounds.max[1],
                bounds.max[2],
                bounds.max[3],
            )
        });
        out
    }
}

fn collect_non_empty_node_bounds(core: &RegionTreeCore, out: &mut Vec<Aabb4i>) {
    match &core.kind {
        RegionNodeKind::Empty => {}
        RegionNodeKind::Branch(children) => {
            for child in children {
                collect_non_empty_node_bounds(child, out);
            }
        }
        _ => out.push(core.bounds),
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

fn intersect_bounds(a: Aabb4i, b: Aabb4i) -> Option<Aabb4i> {
    if !a.intersects(&b) {
        return None;
    }
    let intersection = Aabb4i::new(
        [
            a.min[0].max(b.min[0]),
            a.min[1].max(b.min[1]),
            a.min[2].max(b.min[2]),
            a.min[3].max(b.min[3]),
        ],
        [
            a.max[0].min(b.max[0]),
            a.max[1].min(b.max[1]),
            a.max[2].min(b.max[2]),
            a.max[3].min(b.max[3]),
        ],
    );
    if intersection.is_valid() {
        Some(intersection)
    } else {
        None
    }
}

fn subtract_bounds(outer: Aabb4i, inner: Aabb4i) -> Vec<Aabb4i> {
    let Some(inner) = intersect_bounds(outer, inner) else {
        return vec![outer];
    };
    if inner == outer {
        return Vec::new();
    }

    let mut pieces = Vec::with_capacity(8);
    let mut core = outer;
    for axis in 0..4 {
        if core.min[axis] < inner.min[axis] {
            let mut piece = core;
            piece.max[axis] = inner.min[axis] - 1;
            if piece.is_valid() {
                pieces.push(piece);
            }
            core.min[axis] = inner.min[axis];
        }
        if core.max[axis] > inner.max[axis] {
            let mut piece = core;
            piece.min[axis] = inner.max[axis] + 1;
            if piece.is_valid() {
                pieces.push(piece);
            }
            core.max[axis] = inner.max[axis];
        }
    }
    pieces
}

fn subtract_covered_bounds(target: Aabb4i, covered: &[Aabb4i]) -> Vec<Aabb4i> {
    let mut remaining = vec![target];
    for cover in covered {
        let mut next = Vec::new();
        for bounds in remaining {
            next.extend(subtract_bounds(bounds, *cover));
        }
        if next.is_empty() {
            return Vec::new();
        }
        remaining = next;
    }
    remaining
}

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

    pub fn from_save_root(
        root: &Path,
        default_base_kind: BaseWorldKind,
        world_seed: u64,
        procgen_structures: bool,
        blocked_cells: HashSet<crate::server::procgen::StructureCell>,
        now_ms: u64,
    ) -> io::Result<Self> {
        let metadata =
            save_v4::load_or_init_state_metadata(root, default_base_kind, world_seed, now_ms)?;
        let field = LegacyWorldGenerator::from_chunk_payloads(
            metadata.global.base_world_kind.to_runtime(),
            Vec::<([i32; 4], ChunkPayload)>::new(),
            metadata.global.world_seed,
            procgen_structures,
            blocked_cells,
        );
        Ok(Self {
            field,
            dirty_chunks: RegionChunkTree::new(),
            dirty_save_chunks: HashSet::new(),
            save_stream: Some(SaveStreamingState {
                root: root.to_path_buf(),
                index: metadata.index,
                loaded_bounds: Vec::new(),
                next_entity_id: metadata.global.next_entity_id.max(1),
            }),
        })
    }

    fn ensure_persisted_bounds_loaded(&mut self, bounds: Aabb4i) -> io::Result<usize> {
        if !bounds.is_valid() {
            return Ok(0);
        }
        let Some(save_stream) = self.save_stream.as_mut() else {
            return Ok(0);
        };

        let missing = subtract_covered_bounds(bounds, &save_stream.loaded_bounds);
        if missing.is_empty() {
            return Ok(0);
        }

        let mut loaded_payloads = 0usize;
        for missing_bounds in missing {
            let payloads = save_v4::load_world_chunk_payloads_for_bounds_from_index(
                &save_stream.root,
                &save_stream.index,
                missing_bounds,
            )?;
            loaded_payloads = loaded_payloads.saturating_add(payloads.len());
            let _ = self.field.apply_chunk_payloads(payloads);
            save_stream.loaded_bounds.push(missing_bounds);
        }
        self.field.clear_dirty();
        Ok(loaded_payloads)
    }

    pub fn prepare_query_bounds(&mut self, bounds: Aabb4i) -> io::Result<usize> {
        self.ensure_persisted_bounds_loaded(bounds)
    }

    pub fn persisted_next_entity_id(&self) -> u64 {
        self.save_stream
            .as_ref()
            .map(|stream| stream.next_entity_id.max(1))
            .unwrap_or(1)
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
        self.dirty_save_chunks.clear();
    }

    pub fn set_procgen_blocked_cells(
        &mut self,
        blocked_cells: HashSet<crate::server::procgen::StructureCell>,
    ) {
        self.field.set_procgen_blocked_cells(blocked_cells);
    }

    pub fn rebuild_procgen_keepout_from_chunks(&mut self, padding_chunks: i32) -> usize {
        self.field
            .rebuild_procgen_keepout_from_chunks(padding_chunks)
    }

    pub fn prune_virgin_chunks(&mut self) -> usize {
        self.field.prune_virgin_chunks()
    }

    pub fn apply_voxel_edit(
        &mut self,
        position: [i32; 4],
        material: VoxelType,
    ) -> Option<ChunkPos> {
        let (chunk_pos, _) = world_to_chunk(position[0], position[1], position[2], position[3]);
        let chunk_bounds = Aabb4i::new(
            [chunk_pos.x, chunk_pos.y, chunk_pos.z, chunk_pos.w],
            [chunk_pos.x, chunk_pos.y, chunk_pos.z, chunk_pos.w],
        );
        if let Err(error) = self.ensure_persisted_bounds_loaded(chunk_bounds) {
            eprintln!(
                "failed to hydrate persisted world chunk {} {} {} {} before edit: {}",
                chunk_pos.x, chunk_pos.y, chunk_pos.z, chunk_pos.w, error
            );
        }

        let changed = self.field.apply_voxel_edit(position, material);
        if let Some(chunk_pos) = changed {
            self.mark_dirty_chunk(chunk_pos);
            self.dirty_save_chunks
                .insert([chunk_pos.x, chunk_pos.y, chunk_pos.z, chunk_pos.w]);
        }
        changed
    }

    pub fn persist_dirty_overrides(
        &mut self,
        next_entity_id: u64,
        now_ms: u64,
    ) -> io::Result<Option<save_v4::SaveResult>> {
        if self.save_stream.is_none() {
            return Ok(None);
        }
        if self.dirty_save_chunks.is_empty() {
            if let Some(stream) = self.save_stream.as_mut() {
                stream.next_entity_id = stream.next_entity_id.max(next_entity_id).max(1);
            }
            return Ok(None);
        }

        let mut dirty_chunk_positions: Vec<[i32; 4]> =
            self.dirty_save_chunks.iter().copied().collect();
        dirty_chunk_positions.sort_unstable();
        let dirty_chunk_payloads = dirty_chunk_positions
            .into_iter()
            .map(|chunk_pos| (chunk_pos, self.field.chunk_tree().chunk_payload(chunk_pos)))
            .collect::<Vec<_>>();
        let base_world_kind = self.field.base_kind();
        let world_seed = self.field.world_seed();

        let result = {
            let stream = self
                .save_stream
                .as_ref()
                .expect("save_stream should exist while persisting");
            save_v4::save_state_from_chunk_payload_patch(
                &stream.root,
                SaveChunkPayloadPatchRequest {
                    base_world_kind,
                    dirty_chunk_payloads,
                    world_seed,
                    next_entity_id: next_entity_id.max(stream.next_entity_id).max(1),
                    player_entity_hints: None,
                    custom_global_payload: None,
                    now_ms,
                },
            )?
        };

        self.dirty_save_chunks.clear();
        let Some(result) = result else {
            return Ok(None);
        };

        let root = self
            .save_stream
            .as_ref()
            .map(|stream| stream.root.clone())
            .expect("save_stream should exist while refreshing metadata");
        let refreshed = save_v4::load_state_metadata(&root)?;
        if let Some(stream) = self.save_stream.as_mut() {
            stream.index = refreshed.index;
            stream.next_entity_id = refreshed.global.next_entity_id.max(1);
        }
        Ok(Some(result))
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
    fn overlay_dirty_bounds_drain_returns_touched_chunk_once() {
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

        let dirty = overlay.take_dirty_bounds();
        assert_eq!(dirty, vec![Aabb4i::new([0, 0, 0, 0], [0, 0, 0, 0])]);
        assert!(overlay.take_dirty_bounds().is_empty());
    }

    #[test]
    fn overlay_dirty_bounds_drain_tracks_multiple_chunks_sorted() {
        let mut overlay = ServerWorldOverlay::from_chunk_payloads(
            BaseWorldKind::Empty,
            Vec::<([i32; 4], ChunkPayload)>::new(),
            123,
            false,
            HashSet::new(),
        );

        let _ = overlay.apply_voxel_edit([0, 0, 0, 0], VoxelType(1));
        let _ = overlay.apply_voxel_edit([8, 0, 0, 0], VoxelType(1));
        let _ = overlay.apply_voxel_edit([0, 8, 0, 0], VoxelType(1));

        let dirty = overlay.take_dirty_bounds();
        assert_eq!(
            dirty,
            vec![
                Aabb4i::new([0, 0, 0, 0], [0, 0, 0, 0]),
                Aabb4i::new([0, 1, 0, 0], [0, 1, 0, 0]),
                Aabb4i::new([1, 0, 0, 0], [1, 0, 0, 0]),
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
        assert!(overlay.take_dirty_bounds().is_empty());
    }
}
