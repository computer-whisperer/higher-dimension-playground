use crate::save_v4::{self, SaveChunkPayloadPatchRequest};
use crate::shared::chunk_payload::{ChunkArrayData, ChunkPayload};
use crate::shared::region_tree::{
    chunk_key_from_chunk_pos, RegionChunkTree, RegionNodeKind, RegionTreeCore,
};
use crate::shared::spatial::Aabb4i;
use crate::shared::voxel::{world_to_chunk, BaseWorldKind, ChunkPos, VoxelType, CHUNK_VOLUME};
use std::collections::HashSet;
use std::io;
use std::path::{Path, PathBuf};
use std::sync::Arc;

mod flat_world_generator;
mod massive_platforms_world_generator;

pub use flat_world_generator::FlatWorldGenerator;
pub use massive_platforms_world_generator::MassivePlatformsWorldGenerator;

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
pub enum ServerWorldField {
    Flat(FlatWorldGenerator),
    MassivePlatforms(MassivePlatformsWorldGenerator),
}

impl WorldField for ServerWorldField {
    fn query_region_core(&self, query: QueryVolume, detail: QueryDetail) -> Arc<RegionTreeCore> {
        match self {
            ServerWorldField::Flat(field) => field.query_region_core(query, detail),
            ServerWorldField::MassivePlatforms(field) => field.query_region_core(query, detail),
        }
    }
}

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
    // Authoritative runtime override tree: explicit deltas over virgin world queries.
    override_chunks: RegionChunkTree,
    // Dirty tracking for replication fanout.
    dirty_chunks: RegionChunkTree,
    // Dirty tracking for save patch writes.
    dirty_save_chunks: HashSet<[i32; 4]>,
    save_stream: Option<SaveStreamingState>,
    base_world_kind: BaseWorldKind,
    world_seed: u64,
}

impl<F> PassthroughWorldOverlay<F> {
    pub fn new(field: F, base_world_kind: BaseWorldKind, world_seed: u64) -> Self {
        Self {
            field,
            override_chunks: RegionChunkTree::new(),
            dirty_chunks: RegionChunkTree::new(),
            dirty_save_chunks: HashSet::new(),
            save_stream: None,
            base_world_kind,
            world_seed,
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
        let bounds = query.bounds;
        if !bounds.is_valid() {
            return Arc::new(RegionTreeCore {
                bounds,
                kind: RegionNodeKind::Empty,
                generator_version_hash: 0,
            });
        }

        // Compose authoritative runtime world as:
        // virgin field query + explicit override chunks.
        let base_core = self.field.query_region_core(query, detail);
        let compose_bounds = union_bounds(bounds, base_core.bounds);
        let mut composed = RegionChunkTree::new();
        let _ = composed.splice_non_empty_core_in_bounds(base_core.bounds, base_core.as_ref());
        let override_core = self.override_chunks.slice_core_in_bounds(compose_bounds);
        let _ = composed.overlay_core_in_bounds(compose_bounds, &override_core);
        Arc::new(composed.root().cloned().unwrap_or(RegionTreeCore {
            bounds: compose_bounds,
            kind: RegionNodeKind::Empty,
            generator_version_hash: 0,
        }))
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

fn union_bounds(a: Aabb4i, b: Aabb4i) -> Aabb4i {
    Aabb4i::new(
        [
            a.min[0].min(b.min[0]),
            a.min[1].min(b.min[1]),
            a.min[2].min(b.min[2]),
            a.min[3].min(b.min[3]),
        ],
        [
            a.max[0].max(b.max[0]),
            a.max[1].max(b.max[1]),
            a.max[2].max(b.max[2]),
            a.max[3].max(b.max[3]),
        ],
    )
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

fn build_server_world_field(
    base_kind: BaseWorldKind,
    world_seed: u64,
    procgen_structures: bool,
    blocked_cells: HashSet<crate::server::procgen::StructureCell>,
) -> ServerWorldField {
    match base_kind {
        BaseWorldKind::MassivePlatforms { .. } => {
            ServerWorldField::MassivePlatforms(MassivePlatformsWorldGenerator::from_chunk_payloads(
                base_kind,
                Vec::<([i32; 4], ChunkPayload)>::new(),
                world_seed,
                procgen_structures,
                blocked_cells,
            ))
        }
        BaseWorldKind::FlatFloor { .. } | BaseWorldKind::Empty => {
            ServerWorldField::Flat(FlatWorldGenerator::from_chunk_payloads(
                base_kind,
                Vec::<([i32; 4], ChunkPayload)>::new(),
                world_seed,
                procgen_structures,
                blocked_cells,
            ))
        }
    }
}

impl PassthroughWorldOverlay<ServerWorldField> {
    pub fn from_chunk_payloads(
        base_kind: BaseWorldKind,
        chunk_payloads: impl IntoIterator<Item = ([i32; 4], ChunkPayload)>,
        world_seed: u64,
        procgen_structures: bool,
        blocked_cells: HashSet<crate::server::procgen::StructureCell>,
    ) -> Self {
        let field =
            build_server_world_field(base_kind, world_seed, procgen_structures, blocked_cells);
        Self {
            field,
            override_chunks: RegionChunkTree::from_chunks(chunk_payloads),
            dirty_chunks: RegionChunkTree::new(),
            dirty_save_chunks: HashSet::new(),
            save_stream: None,
            base_world_kind: base_kind,
            world_seed,
        }
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
        let base_world_kind = metadata.global.base_world_kind.to_runtime();
        let runtime_world_seed = metadata.global.world_seed;
        let field = build_server_world_field(
            base_world_kind,
            runtime_world_seed,
            procgen_structures,
            blocked_cells,
        );
        Ok(Self {
            field,
            override_chunks: RegionChunkTree::new(),
            dirty_chunks: RegionChunkTree::new(),
            dirty_save_chunks: HashSet::new(),
            save_stream: Some(SaveStreamingState {
                root: root.to_path_buf(),
                index: metadata.index,
                loaded_bounds: Vec::new(),
                next_entity_id: metadata.global.next_entity_id.max(1),
            }),
            base_world_kind,
            world_seed: runtime_world_seed,
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
            for (chunk_key, payload) in payloads {
                let _ = self.override_chunks.set_chunk(chunk_key, Some(payload));
            }
            save_stream.loaded_bounds.push(missing_bounds);
        }
        Ok(loaded_payloads)
    }

    fn query_virgin_chunk_payload(&self, chunk_pos: ChunkPos) -> Option<ChunkPayload> {
        let key = chunk_key_from_chunk_pos(chunk_pos);
        let bounds = Aabb4i::new(key, key);
        let core = self
            .field
            .query_region_core(QueryVolume { bounds }, QueryDetail::Exact);
        chunk_payload_from_core(core.as_ref(), key)
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
        self.world_seed
    }

    pub fn non_empty_chunk_count(&self) -> usize {
        self.override_chunks.non_empty_chunk_count()
    }

    pub fn clear_dirty(&mut self) {
        self.clear_dirty_chunks();
        self.dirty_save_chunks.clear();
    }

    pub fn apply_voxel_edit(
        &mut self,
        position: [i32; 4],
        material: VoxelType,
    ) -> Option<ChunkPos> {
        let (chunk_pos, voxel_index) =
            world_to_chunk(position[0], position[1], position[2], position[3]);
        let chunk_key = chunk_key_from_chunk_pos(chunk_pos);
        let chunk_bounds = Aabb4i::new(chunk_key, chunk_key);
        if let Err(error) = self.ensure_persisted_bounds_loaded(chunk_bounds) {
            eprintln!(
                "failed to hydrate persisted world chunk {} {} {} {} before edit: {}",
                chunk_pos.x, chunk_pos.y, chunk_pos.z, chunk_pos.w, error
            );
        }

        let override_payload = self.override_chunks.chunk_payload(chunk_key);
        let virgin_payload = self.query_virgin_chunk_payload(chunk_pos);
        let mut working_chunk = override_payload
            .as_ref()
            .and_then(chunk_from_payload)
            .or_else(|| virgin_payload.as_ref().and_then(chunk_from_payload))
            .unwrap_or_else(empty_dense_chunk);

        if working_chunk[voxel_index] == material {
            return None;
        }

        working_chunk[voxel_index] = material;
        let should_remove_override =
            if let Some(virgin_chunk) = virgin_payload.as_ref().and_then(chunk_from_payload) {
                chunks_equal(&working_chunk, &virgin_chunk)
            } else {
                virgin_payload.is_none() && dense_chunk_is_empty(&working_chunk)
            };

        let desired_payload = if should_remove_override {
            None
        } else {
            Some(payload_from_chunk_compact(&working_chunk))
        };
        let previous_payload = self.override_chunks.chunk_payload(chunk_key);
        let changed_by_api = self
            .override_chunks
            .set_chunk(chunk_key, desired_payload.clone());
        let current_payload = self.override_chunks.chunk_payload(chunk_key);
        let mut changed = changed_by_api || previous_payload != current_payload;
        if !changed {
            if current_payload != desired_payload {
                let mut repair_tree = RegionChunkTree::new();
                if let Some(payload) = desired_payload.clone() {
                    let _ = repair_tree.set_chunk(chunk_key, Some(payload));
                }
                let repair_core = repair_tree.root().cloned().unwrap_or(RegionTreeCore {
                    bounds: chunk_bounds,
                    kind: RegionNodeKind::Empty,
                    generator_version_hash: 0,
                });
                if self
                    .override_chunks
                    .splice_non_empty_core_in_bounds(chunk_bounds, &repair_core)
                    .is_some()
                {
                    changed = true;
                    eprintln!(
                        "[server-world-repair] forced override chunk repair at {:?} (expected={:?} current={:?})",
                        chunk_key, desired_payload, current_payload
                    );
                }
            }
        }

        if !changed {
            return None;
        }

        self.mark_dirty_chunk(chunk_pos);
        self.dirty_save_chunks.insert(chunk_key);
        Some(chunk_pos)
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
            .map(|chunk_pos| (chunk_pos, self.override_chunks.chunk_payload(chunk_pos)))
            .collect::<Vec<_>>();

        let result = {
            let stream = self
                .save_stream
                .as_ref()
                .expect("save_stream should exist while persisting");
            save_v4::save_state_from_chunk_payload_patch(
                &stream.root,
                SaveChunkPayloadPatchRequest {
                    base_world_kind: self.base_world_kind,
                    dirty_chunk_payloads,
                    world_seed: self.world_seed,
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
        self.base_world_kind = refreshed.global.base_world_kind.to_runtime();
        self.world_seed = refreshed.global.world_seed;
        Ok(Some(result))
    }

    pub fn chunk_at(&self, chunk_pos: ChunkPos) -> Option<[VoxelType; CHUNK_VOLUME]> {
        self.override_chunks
            .chunk_payload(chunk_key_from_chunk_pos(chunk_pos))
            .and_then(|payload| chunk_from_payload(&payload))
    }

    pub fn effective_chunk(
        &self,
        chunk_pos: ChunkPos,
        preserve_explicit_empty_chunk: bool,
    ) -> Option<[VoxelType; CHUNK_VOLUME]> {
        if let Some(payload) = self
            .override_chunks
            .chunk_payload(chunk_key_from_chunk_pos(chunk_pos))
        {
            return payload_to_effective_chunk(payload, preserve_explicit_empty_chunk);
        }
        let virgin_payload = self.query_virgin_chunk_payload(chunk_pos)?;
        payload_to_effective_chunk(virgin_payload, preserve_explicit_empty_chunk)
    }
}

fn chunk_payload_from_core(core: &RegionTreeCore, key_pos: [i32; 4]) -> Option<ChunkPayload> {
    if !core.bounds.contains_chunk(key_pos) {
        return None;
    }

    match &core.kind {
        RegionNodeKind::Empty => None,
        RegionNodeKind::Uniform(material) => Some(ChunkPayload::Uniform(*material)),
        RegionNodeKind::ProceduralRef(_) => None,
        RegionNodeKind::ChunkArray(chunk_array) => chunk_array_payload_at(chunk_array, key_pos),
        RegionNodeKind::Branch(children) => children
            .iter()
            .find(|child| child.bounds.contains_chunk(key_pos))
            .and_then(|child| chunk_payload_from_core(child, key_pos)),
    }
}

fn chunk_array_payload_at(chunk_array: &ChunkArrayData, key_pos: [i32; 4]) -> Option<ChunkPayload> {
    if !chunk_array.bounds.contains_chunk(key_pos) {
        return None;
    }
    let dense_indices = chunk_array.decode_dense_indices().ok()?;
    let extents = chunk_array.bounds.chunk_extents()?;
    let local = [
        (key_pos[0] - chunk_array.bounds.min[0]) as usize,
        (key_pos[1] - chunk_array.bounds.min[1]) as usize,
        (key_pos[2] - chunk_array.bounds.min[2]) as usize,
        (key_pos[3] - chunk_array.bounds.min[3]) as usize,
    ];
    let linear = linear_cell_index(local, extents);
    let palette_idx = *dense_indices.get(linear)? as usize;
    chunk_array.chunk_palette.get(palette_idx).cloned()
}

fn linear_cell_index(coords: [usize; 4], dims: [usize; 4]) -> usize {
    coords[0] + dims[0] * (coords[1] + dims[1] * (coords[2] + dims[2] * coords[3]))
}

fn payload_to_effective_chunk(
    payload: ChunkPayload,
    preserve_explicit_empty_chunk: bool,
) -> Option<[VoxelType; CHUNK_VOLUME]> {
    let chunk = chunk_from_payload(&payload)?;
    if preserve_explicit_empty_chunk || !dense_chunk_is_empty(&chunk) {
        Some(chunk)
    } else {
        None
    }
}

fn payload_from_chunk_compact(chunk: &[VoxelType; CHUNK_VOLUME]) -> ChunkPayload {
    let materials: Vec<u16> = chunk.iter().map(|voxel| u16::from(voxel.0)).collect();
    ChunkPayload::from_dense_materials_compact(&materials)
        .unwrap_or(ChunkPayload::Dense16 { materials })
}

fn chunk_from_payload(payload: &ChunkPayload) -> Option<[VoxelType; CHUNK_VOLUME]> {
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

fn empty_dense_chunk() -> [VoxelType; CHUNK_VOLUME] {
    [VoxelType::AIR; CHUNK_VOLUME]
}

fn dense_chunk_is_empty(chunk: &[VoxelType; CHUNK_VOLUME]) -> bool {
    chunk.iter().all(|voxel| voxel.is_air())
}

fn chunks_equal(a: &[VoxelType; CHUNK_VOLUME], b: &[VoxelType; CHUNK_VOLUME]) -> bool {
    a[..] == b[..]
}

pub type ServerWorldOverlay = PassthroughWorldOverlay<ServerWorldField>;

#[cfg(test)]
mod tests {
    use super::*;
    use crate::shared::voxel::BaseWorldKind;

    fn dense_materials_from_core_chunk(core: &RegionTreeCore, chunk_key: [i32; 4]) -> Vec<u16> {
        let Some(payload) = chunk_payload_from_core(core, chunk_key) else {
            return vec![0u16; CHUNK_VOLUME];
        };
        let Ok(materials) = payload.dense_materials() else {
            return vec![0u16; CHUNK_VOLUME];
        };
        if materials.len() == CHUNK_VOLUME {
            materials
        } else {
            vec![0u16; CHUNK_VOLUME]
        }
    }

    fn sample_virgin_chunk_dense(
        base_kind: BaseWorldKind,
        world_seed: u64,
        procgen_structures: bool,
        chunk_key: [i32; 4],
    ) -> Vec<u16> {
        sample_virgin_chunk_dense_with_query_bounds(
            base_kind,
            world_seed,
            procgen_structures,
            chunk_key,
            Aabb4i::new(chunk_key, chunk_key),
        )
    }

    fn sample_virgin_chunk_dense_with_query_bounds(
        base_kind: BaseWorldKind,
        world_seed: u64,
        procgen_structures: bool,
        chunk_key: [i32; 4],
        query_bounds: Aabb4i,
    ) -> Vec<u16> {
        let field =
            build_server_world_field(base_kind, world_seed, procgen_structures, HashSet::new());
        assert!(query_bounds.contains_chunk(chunk_key));
        let core = field.query_region_core(
            QueryVolume {
                bounds: query_bounds,
            },
            QueryDetail::Exact,
        );
        dense_materials_from_core_chunk(core.as_ref(), chunk_key)
    }

    fn collect_chunk_keys(bounds_list: &[Aabb4i]) -> Vec<[i32; 4]> {
        let mut keys = Vec::new();
        for bounds in bounds_list {
            for w in bounds.min[3]..=bounds.max[3] {
                for z in bounds.min[2]..=bounds.max[2] {
                    for y in bounds.min[1]..=bounds.max[1] {
                        for x in bounds.min[0]..=bounds.max[0] {
                            keys.push([x, y, z, w]);
                        }
                    }
                }
            }
        }
        keys.sort_unstable();
        keys.dedup();
        keys
    }

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
            collect_chunk_keys(&dirty),
            vec![[0, 0, 0, 0], [0, 1, 0, 0], [1, 0, 0, 0]]
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

    #[test]
    fn overlay_edit_does_not_mutate_virgin_field() {
        let mut overlay = ServerWorldOverlay::from_chunk_payloads(
            BaseWorldKind::FlatFloor {
                material: VoxelType(11),
            },
            Vec::<([i32; 4], ChunkPayload)>::new(),
            777,
            false,
            HashSet::new(),
        );

        let (chunk_pos, voxel_idx) = world_to_chunk(0, -1, 0, 0);
        let chunk_key = chunk_key_from_chunk_pos(chunk_pos);
        let bounds = Aabb4i::new(chunk_key, chunk_key);

        let virgin_before = overlay
            .field()
            .query_region_core(QueryVolume { bounds }, QueryDetail::Exact);
        let virgin_payload_before =
            chunk_payload_from_core(virgin_before.as_ref(), chunk_key).expect("virgin payload");
        let virgin_chunk_before = chunk_from_payload(&virgin_payload_before).expect("virgin chunk");
        assert!(virgin_chunk_before[voxel_idx].is_solid());

        let changed = overlay.apply_voxel_edit([0, -1, 0, 0], VoxelType::AIR);
        assert_eq!(changed, Some(chunk_pos));

        let effective_after = overlay
            .effective_chunk(chunk_pos, true)
            .expect("effective override chunk");
        assert_eq!(effective_after[voxel_idx], VoxelType::AIR);

        let virgin_after = overlay
            .field()
            .query_region_core(QueryVolume { bounds }, QueryDetail::Exact);
        let virgin_payload_after =
            chunk_payload_from_core(virgin_after.as_ref(), chunk_key).expect("virgin payload");
        let virgin_chunk_after = chunk_from_payload(&virgin_payload_after).expect("virgin chunk");
        assert!(virgin_chunk_after[voxel_idx].is_solid());
    }

    #[test]
    fn query_region_core_applies_explicit_empty_override_over_virgin_content() {
        let (chunk_pos, voxel_idx) = world_to_chunk(0, -1, 0, 0);
        let chunk_key = chunk_key_from_chunk_pos(chunk_pos);
        let bounds = Aabb4i::new(chunk_key, chunk_key);
        let mut overlay = ServerWorldOverlay::from_chunk_payloads(
            BaseWorldKind::FlatFloor {
                material: VoxelType(11),
            },
            vec![(chunk_key, ChunkPayload::Uniform(0))],
            991,
            false,
            HashSet::new(),
        );

        let composed = overlay.query_region_core(QueryVolume { bounds }, QueryDetail::Exact);
        let composed_payload =
            chunk_payload_from_core(composed.as_ref(), chunk_key).expect("composed payload");
        let composed_chunk = chunk_from_payload(&composed_payload).expect("composed chunk");
        assert!(composed_chunk[voxel_idx].is_air());

        let virgin = overlay
            .field()
            .query_region_core(QueryVolume { bounds }, QueryDetail::Exact);
        let virgin_payload =
            chunk_payload_from_core(virgin.as_ref(), chunk_key).expect("virgin payload");
        let virgin_chunk = chunk_from_payload(&virgin_payload).expect("virgin chunk");
        assert!(virgin_chunk[voxel_idx].is_solid());

        overlay.clear_dirty();
    }

    #[test]
    fn overlay_query_preserves_generator_leaf_expansion_beyond_request_bounds() {
        let overlay = ServerWorldOverlay::from_chunk_payloads(
            BaseWorldKind::MassivePlatforms {
                material: VoxelType(11),
            },
            Vec::<([i32; 4], ChunkPayload)>::new(),
            1337,
            false,
            HashSet::new(),
        );
        let query_bounds = Aabb4i::new([0, -1, 0, 0], [0, -1, 0, 0]);
        let core = overlay.query_region_core(
            QueryVolume {
                bounds: query_bounds,
            },
            QueryDetail::Exact,
        );
        assert!(core.bounds.contains_chunk(query_bounds.min));
        assert!(core.bounds.contains_chunk(query_bounds.max));
        assert_ne!(core.bounds, query_bounds);
    }

    #[test]
    fn overlay_edit_is_visible_when_generator_returns_expanded_platform_leaf() {
        let mut overlay = ServerWorldOverlay::from_chunk_payloads(
            BaseWorldKind::MassivePlatforms {
                material: VoxelType(11),
            },
            Vec::<([i32; 4], ChunkPayload)>::new(),
            1337,
            true,
            HashSet::new(),
        );
        let edit_pos = [0, -1, 0, 0];
        let (chunk_pos, voxel_idx) =
            world_to_chunk(edit_pos[0], edit_pos[1], edit_pos[2], edit_pos[3]);
        let chunk_key = chunk_key_from_chunk_pos(chunk_pos);
        let chunk_bounds = Aabb4i::new(chunk_key, chunk_key);
        assert_eq!(
            overlay.apply_voxel_edit(edit_pos, VoxelType(5)),
            Some(chunk_pos)
        );

        let queried = overlay.query_region_core(
            QueryVolume {
                bounds: chunk_bounds,
            },
            QueryDetail::Exact,
        );
        if let Some(payload) = chunk_payload_from_core(queried.as_ref(), chunk_key) {
            let chunk = chunk_from_payload(&payload).expect("edited dense chunk");
            assert_eq!(
                chunk[voxel_idx],
                VoxelType(5),
                "expanded-bounds overlay query returned wrong edited voxel value",
            );
        }
    }

    #[test]
    fn virgin_world_generator_chunk_sampling_is_deterministic_for_observed_coords() {
        let observed_chunks = [
            [-14, -1, -22, -19],
            [-10, -2, -4, 0],
            [-12, -1, 0, 7],
            [-18, -14, -24, 22],
            [17, -26, -20, 23],
            [5, -2, 10, 3],
        ];
        let world_kinds = [
            BaseWorldKind::FlatFloor {
                material: VoxelType(11),
            },
            BaseWorldKind::MassivePlatforms {
                material: VoxelType(11),
            },
        ];

        for base_kind in world_kinds {
            for chunk_key in observed_chunks {
                let baseline = sample_virgin_chunk_dense(base_kind, 0xD1A6_2026, true, chunk_key);
                for sample_idx in 0..16 {
                    let sample = sample_virgin_chunk_dense(base_kind, 0xD1A6_2026, true, chunk_key);
                    assert_eq!(
                        sample, baseline,
                        "virgin generator changed output for base_kind={base_kind:?} chunk={chunk_key:?} sample_idx={sample_idx}",
                    );
                }
            }
        }
    }

    #[test]
    fn virgin_world_generator_chunk_sampling_is_query_volume_invariant() {
        let observed_chunks = [
            [-14, -1, -22, -19],
            [-10, -2, -4, 0],
            [-12, -1, 0, 7],
            [-18, -14, -24, 22],
            [17, -26, -20, 23],
            [5, -2, 10, 3],
        ];
        let world_kinds = [
            BaseWorldKind::FlatFloor {
                material: VoxelType(11),
            },
            BaseWorldKind::MassivePlatforms {
                material: VoxelType(11),
            },
        ];
        let query_radii = [0i32, 1, 2, 4, 8, 16];

        for base_kind in world_kinds {
            for chunk_key in observed_chunks {
                let baseline = sample_virgin_chunk_dense(base_kind, 0xD1A6_2026, true, chunk_key);
                for radius in query_radii {
                    let query_bounds = Aabb4i::new(
                        [
                            chunk_key[0] - radius,
                            chunk_key[1] - radius,
                            chunk_key[2] - radius,
                            chunk_key[3] - radius,
                        ],
                        [
                            chunk_key[0] + radius,
                            chunk_key[1] + radius,
                            chunk_key[2] + radius,
                            chunk_key[3] + radius,
                        ],
                    );
                    let sample = sample_virgin_chunk_dense_with_query_bounds(
                        base_kind,
                        0xD1A6_2026,
                        true,
                        chunk_key,
                        query_bounds,
                    );
                    assert_eq!(
                        sample, baseline,
                        "virgin generator changed per query volume: base_kind={base_kind:?} chunk={chunk_key:?} query_bounds={:?}->{:?}",
                        query_bounds.min,
                        query_bounds.max
                    );
                }
            }
        }
    }
}
