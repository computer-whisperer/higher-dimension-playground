use crate::save_v4::{self, SaveChunkPayloadPatchRequest};
use crate::shared::chunk_payload::{ChunkArrayData, ChunkPayload, ResolvedChunkPayload};
use crate::shared::protocol::WorldBounds;
use crate::shared::region_tree::{
    ChunkKey, RegionChunkTree, RegionNodeKind, RegionTreeCore,
};
use crate::shared::spatial::{Aabb4i, ChunkCoord};
use crate::shared::voxel::{
    world_to_chunk, BaseWorldKind, BlockData, CHUNK_VOLUME,
};
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
    dirty_save_chunks: HashSet<ChunkKey>,
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

    pub fn world_bounds(&self) -> WorldBounds {
        match self.base_world_kind {
            BaseWorldKind::FlatFloor { .. } => WorldBounds {
                min: [None, Some(-4.0), None, None],
                max: [None; 4],
            },
            _ => WorldBounds::default(),
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

    pub fn mark_dirty_chunk(&mut self, chunk_key: ChunkKey) {
        let _ = self.dirty_chunks.set_chunk(
            chunk_key,
            Some(ResolvedChunkPayload::uniform(BlockData::simple(0, 1))),
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
        //
        // NOTE: `base_core.bounds` is intentionally allowed to exceed
        // `query.bounds`. This preserves canonical large leaves (for example,
        // wide Uniform spans) across repeated overlapping queries so client
        // cache merges naturally de-fragment instead of accumulating clipped
        // fragments.
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

    let one = ChunkCoord::from_num(1);
    let mut pieces = Vec::with_capacity(8);
    let mut core = outer;
    for axis in 0..4 {
        if core.min[axis] < inner.min[axis] {
            let mut piece = core;
            piece.max[axis] = inner.min[axis] - one;
            if piece.is_valid() {
                pieces.push(piece);
            }
            core.min[axis] = inner.min[axis];
        }
        if core.max[axis] > inner.max[axis] {
            let mut piece = core;
            piece.min[axis] = inner.max[axis] + one;
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
        chunk_payloads: impl IntoIterator<Item = ([i32; 4], ResolvedChunkPayload)>,
        world_seed: u64,
        procgen_structures: bool,
        blocked_cells: HashSet<crate::server::procgen::StructureCell>,
    ) -> Self {
        let field =
            build_server_world_field(base_kind.clone(), world_seed, procgen_structures, blocked_cells);
        Self {
            field,
            override_chunks: RegionChunkTree::from_chunks(
                chunk_payloads.into_iter().map(|(pos, payload)| {
                    (pos.map(ChunkCoord::from_num), payload)
                }),
            ),
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
            base_world_kind.clone(),
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

        let mut loaded_regions = 0usize;
        for missing_bounds in missing {
            let core = save_v4::load_world_subtree_core_for_bounds_from_index(
                &save_stream.root,
                &save_stream.index,
                missing_bounds,
            )?;
            let _ = self
                .override_chunks
                .splice_core_in_bounds(missing_bounds, &core);
            loaded_regions = loaded_regions.saturating_add(1);
            save_stream.loaded_bounds.push(missing_bounds);
        }
        Ok(loaded_regions)
    }

    fn query_virgin_chunk_payload(&self, chunk_key: ChunkKey) -> Option<ResolvedChunkPayload> {
        let bounds = Aabb4i::new(chunk_key, chunk_key);
        let core = self
            .field
            .query_region_core(QueryVolume { bounds }, QueryDetail::Exact);
        chunk_payload_from_core(core.as_ref(), chunk_key)
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
        block: BlockData,
    ) -> Option<ChunkKey> {
        let (chunk_key, voxel_index) =
            world_to_chunk(position[0], position[1], position[2], position[3]);
        let chunk_bounds = Aabb4i::new(chunk_key, chunk_key);
        if let Err(error) = self.ensure_persisted_bounds_loaded(chunk_bounds) {
            eprintln!(
                "failed to hydrate persisted world chunk {:?} before edit: {}",
                chunk_key, error
            );
        }

        let override_payload = self.override_chunks.chunk_payload(chunk_key);
        let virgin_payload = self.query_virgin_chunk_payload(chunk_key);
        let current = override_payload.as_ref().or(virgin_payload.as_ref());

        let current_block = current
            .map(|r| r.block_at(voxel_index))
            .unwrap_or(BlockData::AIR);
        if current_block == block {
            return None;
        }

        let mut blocks = current
            .map(|r| r.dense_blocks())
            .unwrap_or_else(|| vec![BlockData::AIR; CHUNK_VOLUME]);
        blocks[voxel_index] = block;

        let should_remove_override = if let Some(virgin) = virgin_payload.as_ref() {
            blocks == virgin.dense_blocks()
        } else {
            blocks.iter().all(|b| b.is_air())
        };

        let desired_payload = if should_remove_override {
            None
        } else {
            ResolvedChunkPayload::from_dense_blocks(&blocks).ok()
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

        self.mark_dirty_chunk(chunk_key);
        self.dirty_save_chunks.insert(chunk_key);
        Some(chunk_key)
    }

    /// Edit a voxel at a specific scale. For `scale_exp = 0`, delegates to
    /// [`apply_voxel_edit`]. For non-zero scales, uses the scale-aware mutation
    /// API to insert/modify chunks in the override tree.
    pub fn apply_voxel_edit_at_scale(
        &mut self,
        position: [i32; 4],
        block: BlockData,
        scale_exp: i8,
    ) -> Option<ChunkKey> {
        if scale_exp == 0 {
            return self.apply_voxel_edit(position, block);
        }

        let (chunk_key, voxel_index) = crate::shared::voxel::world_to_chunk_at_scale(
            position[0],
            position[1],
            position[2],
            position[3],
            scale_exp,
        );

        let current_payload = self.override_chunks.chunk_payload(chunk_key);
        let current_block = current_payload
            .as_ref()
            .map(|r| r.block_at(voxel_index))
            .unwrap_or(BlockData::AIR);
        if current_block == block {
            return None;
        }

        let mut blocks = current_payload
            .map(|r| r.dense_blocks())
            .unwrap_or_else(|| vec![BlockData::AIR; crate::shared::voxel::CHUNK_VOLUME]);
        blocks[voxel_index] = block;

        let desired_payload = if blocks.iter().all(|b| b.is_air()) {
            None
        } else {
            ResolvedChunkPayload::from_dense_blocks(&blocks).ok()
        };

        let changed = self
            .override_chunks
            .set_chunk_at_scale(chunk_key, desired_payload, scale_exp);
        if !changed {
            return None;
        }

        self.mark_dirty_chunk(chunk_key);
        self.dirty_save_chunks.insert(chunk_key);
        Some(chunk_key)
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

        let mut dirty_keys: Vec<ChunkKey> =
            self.dirty_save_chunks.iter().copied().collect();
        dirty_keys.sort_unstable();
        let dirty_chunk_payloads = dirty_keys
            .into_iter()
            .map(|key| {
                let resolved = self.override_chunks.chunk_payload(key);
                (key, resolved)
            })
            .collect::<Vec<_>>();

        let result = {
            let stream = self
                .save_stream
                .as_ref()
                .expect("save_stream should exist while persisting");
            save_v4::save_state_from_chunk_payload_patch(
                &stream.root,
                SaveChunkPayloadPatchRequest {
                    base_world_kind: self.base_world_kind.clone(),
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

    pub fn chunk_at(&self, chunk_key: ChunkKey) -> Option<ResolvedChunkPayload> {
        self.override_chunks
            .chunk_payload(chunk_key)
    }

    pub fn effective_chunk(
        &self,
        chunk_key: ChunkKey,
        preserve_explicit_empty_chunk: bool,
    ) -> Option<ResolvedChunkPayload> {
        if let Some(payload) = self
            .override_chunks
            .chunk_payload(chunk_key)
        {
            return resolved_to_effective(payload, preserve_explicit_empty_chunk);
        }
        let virgin_payload = self.query_virgin_chunk_payload(chunk_key)?;
        resolved_to_effective(virgin_payload, preserve_explicit_empty_chunk)
    }
}

fn chunk_payload_from_core(
    core: &RegionTreeCore,
    key_pos: ChunkKey,
) -> Option<ResolvedChunkPayload> {
    if !core.bounds.contains_chunk(key_pos) {
        return None;
    }

    match &core.kind {
        RegionNodeKind::Empty => None,
        RegionNodeKind::Uniform(block) => Some(ResolvedChunkPayload::uniform(block.clone())),
        RegionNodeKind::ProceduralRef(_) => None,
        RegionNodeKind::ChunkArray(chunk_array) => {
            let payload = chunk_array_payload_at(chunk_array, key_pos)?;
            Some(ResolvedChunkPayload {
                payload,
                block_palette: chunk_array.block_palette.clone(),
            })
        }
        RegionNodeKind::Branch(children) => children
            .iter()
            .find(|child| child.bounds.contains_chunk(key_pos))
            .and_then(|child| chunk_payload_from_core(child, key_pos)),
    }
}

fn chunk_array_payload_at(chunk_array: &ChunkArrayData, key_pos: ChunkKey) -> Option<ChunkPayload> {
    if !chunk_array.bounds.contains_chunk(key_pos) {
        return None;
    }
    let dense_indices = chunk_array.decode_dense_indices().ok()?;
    let extents = chunk_array.bounds.chunk_extents()?;
    let local = [
        (key_pos[0] - chunk_array.bounds.min[0]).to_num::<i32>() as usize,
        (key_pos[1] - chunk_array.bounds.min[1]).to_num::<i32>() as usize,
        (key_pos[2] - chunk_array.bounds.min[2]).to_num::<i32>() as usize,
        (key_pos[3] - chunk_array.bounds.min[3]).to_num::<i32>() as usize,
    ];
    let linear = linear_cell_index(local, extents);
    let palette_idx = *dense_indices.get(linear)? as usize;
    chunk_array.chunk_palette.get(palette_idx).cloned()
}

fn linear_cell_index(coords: [usize; 4], dims: [usize; 4]) -> usize {
    coords[0] + dims[0] * (coords[1] + dims[1] * (coords[2] + dims[2] * coords[3]))
}

fn resolved_to_effective(
    resolved: ResolvedChunkPayload,
    preserve_explicit_empty_chunk: bool,
) -> Option<ResolvedChunkPayload> {
    if preserve_explicit_empty_chunk {
        Some(resolved)
    } else {
        // Filter out all-air payloads
        match &resolved.payload {
            ChunkPayload::Empty => None,
            ChunkPayload::Uniform(idx) => {
                let is_air = resolved
                    .block_palette
                    .get(*idx as usize)
                    .map(|b| b.is_air())
                    .unwrap_or(true);
                if is_air { None } else { Some(resolved) }
            }
            _ => Some(resolved),
        }
    }
}

pub type ServerWorldOverlay = PassthroughWorldOverlay<ServerWorldField>;

#[cfg(test)]
mod tests {
    use super::*;
    use crate::content_registry::ContentRegistry;
    use crate::shared::voxel::BaseWorldKind;

    fn test_registry() -> Arc<ContentRegistry> {
        Arc::new(crate::plugin_loader::create_full_registry())
    }

    fn dense_materials_from_core_chunk(core: &RegionTreeCore, chunk_key: ChunkKey) -> Vec<u16> {
        let reg = test_registry();
        let Some(payload) = chunk_payload_from_core(core, chunk_key) else {
            return vec![0u16; CHUNK_VOLUME];
        };
        let Ok(materials) = payload.payload.dense_materials() else {
            return vec![0u16; CHUNK_VOLUME];
        };
        if materials.len() == CHUNK_VOLUME {
            // Resolve palette indices to material IDs through block_palette
            materials
                .iter()
                .map(|&idx| {
                    let air = BlockData::AIR;
                    let block = payload
                        .block_palette
                        .get(idx as usize)
                        .unwrap_or(&air);
                    reg.block_material_token(block.namespace, block.block_type)
                })
                .collect()
        } else {
            vec![0u16; CHUNK_VOLUME]
        }
    }

    fn sample_virgin_chunk_dense(
        base_kind: &BaseWorldKind,
        world_seed: u64,
        procgen_structures: bool,
        chunk_key: [i32; 4],
    ) -> Vec<u16> {
        let key = chunk_key.map(ChunkCoord::from_num);
        sample_virgin_chunk_dense_with_query_bounds(
            base_kind,
            world_seed,
            procgen_structures,
            key,
            Aabb4i::new(key, key),
        )
    }

    fn sample_virgin_chunk_dense_with_query_bounds(
        base_kind: &BaseWorldKind,
        world_seed: u64,
        procgen_structures: bool,
        chunk_key: ChunkKey,
        query_bounds: Aabb4i,
    ) -> Vec<u16> {
        let field =
            build_server_world_field(base_kind.clone(), world_seed, procgen_structures, HashSet::new());
        assert!(query_bounds.contains_chunk(chunk_key));
        let core = field.query_region_core(
            QueryVolume {
                bounds: query_bounds,
            },
            QueryDetail::Exact,
        );
        dense_materials_from_core_chunk(core.as_ref(), chunk_key)
    }

    fn collect_chunk_keys(bounds_list: &[Aabb4i]) -> Vec<ChunkKey> {
        let mut keys = Vec::new();
        for bounds in bounds_list {
            let (bmin, bmax) = bounds.to_lattice_bounds(0);
            for w in bmin[3]..=bmax[3] {
                for z in bmin[2]..=bmax[2] {
                    for y in bmin[1]..=bmax[1] {
                        for x in bmin[0]..=bmax[0] {
                            keys.push([x, y, z, w].map(ChunkCoord::from_num));
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
            Vec::<([i32; 4], ResolvedChunkPayload)>::new(),
            123,
            false,
            HashSet::new(),
        );

        let changed_a = overlay.apply_voxel_edit([0, 0, 0, 0], BlockData::simple(0, 3));
        let changed_b = overlay.apply_voxel_edit([1, 0, 0, 0], BlockData::simple(0, 4));
        use crate::shared::region_tree::chunk_key_i32;
        assert_eq!(changed_a, Some(chunk_key_i32(0, 0, 0, 0)));
        assert_eq!(changed_b, Some(chunk_key_i32(0, 0, 0, 0)));

        let dirty = overlay.take_dirty_bounds();
        assert_eq!(dirty, vec![Aabb4i::from_i32([0, 0, 0, 0], [0, 0, 0, 0])]);
        assert!(overlay.take_dirty_bounds().is_empty());
    }

    #[test]
    fn overlay_dirty_bounds_drain_tracks_multiple_chunks_sorted() {
        let mut overlay = ServerWorldOverlay::from_chunk_payloads(
            BaseWorldKind::Empty,
            Vec::<([i32; 4], ResolvedChunkPayload)>::new(),
            123,
            false,
            HashSet::new(),
        );

        let _ = overlay.apply_voxel_edit([0, 0, 0, 0], BlockData::simple(0, 1));
        let _ = overlay.apply_voxel_edit([8, 0, 0, 0], BlockData::simple(0, 1));
        let _ = overlay.apply_voxel_edit([0, 8, 0, 0], BlockData::simple(0, 1));

        let dirty = overlay.take_dirty_bounds();
        use crate::shared::region_tree::chunk_key_i32;
        assert_eq!(
            collect_chunk_keys(&dirty),
            vec![chunk_key_i32(0, 0, 0, 0), chunk_key_i32(0, 1, 0, 0), chunk_key_i32(1, 0, 0, 0)]
        );
    }

    #[test]
    fn overlay_clear_dirty_clears_overlay_dirty_chunks() {
        let mut overlay = ServerWorldOverlay::from_chunk_payloads(
            BaseWorldKind::Empty,
            Vec::<([i32; 4], ResolvedChunkPayload)>::new(),
            123,
            false,
            HashSet::new(),
        );

        let _ = overlay.apply_voxel_edit([0, 0, 0, 0], BlockData::simple(0, 5));
        overlay.clear_dirty();
        assert!(overlay.take_dirty_bounds().is_empty());
    }

    #[test]
    fn overlay_edit_does_not_mutate_virgin_field() {
        let mut overlay = ServerWorldOverlay::from_chunk_payloads(
            BaseWorldKind::FlatFloor {
                material: BlockData::simple(0, 11),
            },
            Vec::<([i32; 4], ResolvedChunkPayload)>::new(),
            777,
            false,
            HashSet::new(),
        );

        let (chunk_key, voxel_idx) = world_to_chunk(0, -1, 0, 0);
        let bounds = Aabb4i::new(chunk_key, chunk_key);

        let virgin_before = overlay
            .field()
            .query_region_core(QueryVolume { bounds }, QueryDetail::Exact);
        let virgin_payload_before =
            chunk_payload_from_core(virgin_before.as_ref(), chunk_key).expect("virgin payload");
        assert!(!virgin_payload_before.block_at(voxel_idx).is_air());

        let changed = overlay.apply_voxel_edit([0, -1, 0, 0], BlockData::AIR);
        assert_eq!(changed, Some(chunk_key));

        let effective_after = overlay
            .effective_chunk(chunk_key, true)
            .expect("effective override chunk");
        assert!(effective_after.block_at(voxel_idx).is_air());

        let virgin_after = overlay
            .field()
            .query_region_core(QueryVolume { bounds }, QueryDetail::Exact);
        let virgin_payload_after =
            chunk_payload_from_core(virgin_after.as_ref(), chunk_key).expect("virgin payload");
        assert!(!virgin_payload_after.block_at(voxel_idx).is_air());
    }

    #[test]
    fn query_region_core_applies_explicit_empty_override_over_virgin_content() {
        let (chunk_key, voxel_idx) = world_to_chunk(0, -1, 0, 0);
        let bounds = Aabb4i::new(chunk_key, chunk_key);
        let chunk_key_i32 = [0i32, -1, 0, 0];
        let mut overlay = ServerWorldOverlay::from_chunk_payloads(
            BaseWorldKind::FlatFloor {
                material: BlockData::simple(0, 11),
            },
            vec![(chunk_key_i32, ResolvedChunkPayload::empty())],
            991,
            false,
            HashSet::new(),
        );

        let composed = overlay.query_region_core(QueryVolume { bounds }, QueryDetail::Exact);
        let composed_payload =
            chunk_payload_from_core(composed.as_ref(), chunk_key).expect("composed payload");
        assert!(composed_payload.block_at(voxel_idx).is_air());

        let virgin = overlay
            .field()
            .query_region_core(QueryVolume { bounds }, QueryDetail::Exact);
        let virgin_payload =
            chunk_payload_from_core(virgin.as_ref(), chunk_key).expect("virgin payload");
        assert!(!virgin_payload.block_at(voxel_idx).is_air());

        overlay.clear_dirty();
    }

    #[test]
    fn overlay_query_preserves_generator_leaf_expansion_beyond_request_bounds() {
        let overlay = ServerWorldOverlay::from_chunk_payloads(
            BaseWorldKind::MassivePlatforms {
                material: BlockData::simple(0, 11),
            },
            Vec::<([i32; 4], ResolvedChunkPayload)>::new(),
            1337,
            false,
            HashSet::new(),
        );
        let query_bounds = Aabb4i::from_i32([0, -1, 0, 0], [0, -1, 0, 0]);
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
                material: BlockData::simple(0, 11),
            },
            Vec::<([i32; 4], ResolvedChunkPayload)>::new(),
            1337,
            true,
            HashSet::new(),
        );
        let edit_pos = [0, -1, 0, 0];
        let (chunk_key, voxel_idx) =
            world_to_chunk(edit_pos[0], edit_pos[1], edit_pos[2], edit_pos[3]);
        let chunk_bounds = Aabb4i::new(chunk_key, chunk_key);
        assert_eq!(
            overlay.apply_voxel_edit(edit_pos, BlockData::simple(0, 5)),
            Some(chunk_key)
        );

        let queried = overlay.query_region_core(
            QueryVolume {
                bounds: chunk_bounds,
            },
            QueryDetail::Exact,
        );
        if let Some(payload) = chunk_payload_from_core(queried.as_ref(), chunk_key) {
            let block = payload.block_at(voxel_idx);
            assert_eq!(
                block,
                BlockData::simple(0, 5),
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
                material: BlockData::simple(0, 11),
            },
            BaseWorldKind::MassivePlatforms {
                material: BlockData::simple(0, 11),
            },
        ];

        for base_kind in &world_kinds {
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
                material: BlockData::simple(0, 11),
            },
            BaseWorldKind::MassivePlatforms {
                material: BlockData::simple(0, 11),
            },
        ];
        let query_radii = [0i32, 1, 2, 4, 8, 16];

        for base_kind in &world_kinds {
            for chunk_key in observed_chunks {
                let baseline = sample_virgin_chunk_dense(base_kind, 0xD1A6_2026, true, chunk_key);
                for radius in query_radii {
                    let query_bounds = Aabb4i::from_i32(
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
                    let key = chunk_key.map(ChunkCoord::from_num);
                    let sample = sample_virgin_chunk_dense_with_query_bounds(
                        base_kind,
                        0xD1A6_2026,
                        true,
                        key,
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

    /// Verify that placing the same block as the platform material is correctly
    /// detected as a no-op (no dirty bounds produced).
    #[test]
    fn voxel_edit_same_as_platform_material_is_noop() {
        let platform_material = BlockData::simple(0, 11);
        let mut overlay = ServerWorldOverlay::from_chunk_payloads(
            BaseWorldKind::MassivePlatforms {
                material: platform_material.clone(),
            },
            Vec::<([i32; 4], ResolvedChunkPayload)>::new(),
            1337,
            false,
            HashSet::new(),
        );

        // Edit within the platform, placing the same material.
        let edit_pos = [-1, 0, -6, -4];
        let changed = overlay.apply_voxel_edit(edit_pos, platform_material);
        assert_eq!(changed, None, "placing same block as platform should be no-op");
        assert!(overlay.take_dirty_bounds().is_empty());
    }

    /// Verify that placing a *different* block on the platform succeeds and
    /// produces dirty bounds that a client can receive.
    #[test]
    fn voxel_edit_different_from_platform_material_produces_dirty_bounds() {
        let mut overlay = ServerWorldOverlay::from_chunk_payloads(
            BaseWorldKind::MassivePlatforms {
                material: BlockData::simple(0, 11),
            },
            Vec::<([i32; 4], ResolvedChunkPayload)>::new(),
            1337,
            false,
            HashSet::new(),
        );

        let edit_pos = [-1, 0, -6, -4];
        let (edit_chunk_key, edit_voxel_idx) =
            world_to_chunk(edit_pos[0], edit_pos[1], edit_pos[2], edit_pos[3]);

        // Place a different block.
        let edit_block = BlockData::simple(0, 42);
        let changed = overlay.apply_voxel_edit(edit_pos, edit_block.clone());
        assert_eq!(changed, Some(edit_chunk_key));

        let dirty_bounds = overlay.take_dirty_bounds();
        assert!(!dirty_bounds.is_empty());
    }
}
