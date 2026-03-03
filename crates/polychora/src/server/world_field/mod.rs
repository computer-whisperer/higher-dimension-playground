use crate::save_v4::{self, SaveChunkPayloadPatchRequest};
use crate::shared::chunk_payload::{ChunkArrayData, ChunkPayload, ResolvedChunkPayload};
use crate::shared::protocol::WorldBounds;
use crate::shared::region_tree::{ChunkKey, RegionChunkTree, RegionNodeKind, RegionTreeCore};
use crate::shared::spatial::{lattice_from_fixed, step_for_scale, Aabb4i, ChunkCoord};
use crate::shared::voxel::{
    linear_cell_index, world_to_chunk_at_scale, BaseWorldKind, BlockData, CHUNK_SIZE, CHUNK_VOLUME,
};
use std::collections::{HashMap, HashSet};
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
    // Dirty tracking for save patch writes: ChunkKey → scale_exp.
    dirty_save_chunks: HashMap<ChunkKey, i8>,
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
            dirty_save_chunks: HashMap::new(),
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

    pub fn mark_dirty_chunk(&mut self, chunk_key: ChunkKey, scale_exp: i8) {
        let _ = self.dirty_chunks.set_chunk_at_scale(
            chunk_key,
            Some(ResolvedChunkPayload::uniform(BlockData::simple(0, 1))),
            scale_exp,
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

    let mut pieces = Vec::with_capacity(8);
    let mut core = outer;
    for axis in 0..4 {
        if core.min[axis] < inner.min[axis] {
            let mut piece = core;
            piece.max[axis] = inner.min[axis];
            if piece.is_valid() {
                pieces.push(piece);
            }
            core.min[axis] = inner.min[axis];
        }
        if core.max[axis] > inner.max[axis] {
            let mut piece = core;
            piece.min[axis] = inner.max[axis];
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
        let field = build_server_world_field(
            base_kind.clone(),
            world_seed,
            procgen_structures,
            blocked_cells,
        );
        Self {
            field,
            override_chunks: RegionChunkTree::from_chunks(
                chunk_payloads
                    .into_iter()
                    .map(|(pos, payload)| (pos.map(ChunkCoord::from_num), payload)),
            ),
            dirty_chunks: RegionChunkTree::new(),
            dirty_save_chunks: HashMap::new(),
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
            dirty_save_chunks: HashMap::new(),
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
        let bounds = Aabb4i::chunk_world_bounds(chunk_key, 0);
        let core = self
            .field
            .query_region_core(QueryVolume { bounds }, QueryDetail::Exact);
        chunk_payload_from_core(core.as_ref(), chunk_key)
    }

    /// Query the virgin worldgen data for a chunk at any scale, returning 8^4 dense blocks.
    ///
    /// For `scale_exp == 0`, delegates to [`query_virgin_chunk_payload`].
    /// For other scales, queries the virgin tree for the encompassing spatial region
    /// and resamples by mapping each cell back to its corresponding scale-0 position.
    fn query_virgin_blocks_at_scale(&self, chunk_key: ChunkKey, scale_exp: i8) -> Vec<BlockData> {
        // Fast path for scale 0.
        if scale_exp == 0 {
            return self
                .query_virgin_chunk_payload(chunk_key)
                .map(|p| p.dense_blocks())
                .unwrap_or_else(|| vec![BlockData::AIR; CHUNK_VOLUME]);
        }

        // Compute the global lattice origin of this chunk at its scale.
        let lattice_origin: [i32; 4] =
            std::array::from_fn(|ax| lattice_from_fixed(chunk_key[ax], scale_exp));

        // Each cell's global lattice position at this scale.
        let cell_min: [i64; 4] =
            std::array::from_fn(|ax| lattice_origin[ax] as i64 * CHUNK_SIZE as i64);

        // Compute the world-space integer range (scale-0 lattice positions)
        // that our cells map to.
        let (world_min, world_max) = if scale_exp > 0 {
            let shift = scale_exp as u32;
            let wmin = cell_min.map(|v| (v << shift) as i32);
            let cell_max: [i64; 4] =
                std::array::from_fn(|ax| cell_min[ax] + (CHUNK_SIZE as i64 - 1));
            let wmax = cell_max.map(|v| ((v << shift) + ((1i64 << shift) - 1)) as i32);
            (wmin, wmax)
        } else {
            let shift = (-scale_exp) as u32;
            let wmin = cell_min.map(|v| v.div_euclid(1i64 << shift) as i32);
            let cell_max: [i64; 4] =
                std::array::from_fn(|ax| cell_min[ax] + (CHUNK_SIZE as i64 - 1));
            let wmax = cell_max.map(|v| v.div_euclid(1i64 << shift) as i32);
            (wmin, wmax)
        };

        // Query the virgin tree for the encompassing scale-0 chunk region.
        let cs = CHUNK_SIZE as i32;
        let query_min: [i32; 4] = world_min.map(|v| v.div_euclid(cs));
        let query_max: [i32; 4] = world_max.map(|v| v.div_euclid(cs));
        let query_bounds = Aabb4i::from_lattice_bounds(query_min, query_max, 0);
        let virgin_core = self.field.query_region_core(
            QueryVolume {
                bounds: query_bounds,
            },
            QueryDetail::Exact,
        );

        // Cache resolved payloads per scale-0 chunk to avoid redundant tree lookups.
        let mut payload_cache: HashMap<ChunkKey, Option<ResolvedChunkPayload>> = HashMap::new();
        let mut blocks = vec![BlockData::AIR; CHUNK_VOLUME];

        for lw in 0..CHUNK_SIZE {
            for lz in 0..CHUNK_SIZE {
                for ly in 0..CHUNK_SIZE {
                    for lx in 0..CHUNK_SIZE {
                        let cell_lat: [i64; 4] = [
                            cell_min[0] + lx as i64,
                            cell_min[1] + ly as i64,
                            cell_min[2] + lz as i64,
                            cell_min[3] + lw as i64,
                        ];

                        let step = step_for_scale(scale_exp);
                        let world_pos: [ChunkCoord; 4] =
                            cell_lat.map(|v| ChunkCoord::from_num(v as i32) * step);

                        let (s0_key, s0_idx) = world_to_chunk_at_scale(
                            world_pos[0],
                            world_pos[1],
                            world_pos[2],
                            world_pos[3],
                            0,
                        );

                        let payload = payload_cache.entry(s0_key).or_insert_with(|| {
                            chunk_payload_from_core(virgin_core.as_ref(), s0_key)
                        });

                        let block = payload
                            .as_ref()
                            .map(|p| p.block_at(s0_idx))
                            .unwrap_or(BlockData::AIR);

                        let idx = lw * CHUNK_SIZE * CHUNK_SIZE * CHUNK_SIZE
                            + lz * CHUNK_SIZE * CHUNK_SIZE
                            + ly * CHUNK_SIZE
                            + lx;
                        blocks[idx] = block;
                    }
                }
            }
        }

        blocks
    }

    /// Read the effective (override + virgin) world state for a chunk region.
    ///
    /// Returns `CHUNK_VOLUME` dense blocks at the given chunk position and scale.
    /// Layers override data on top of virgin data by walking the override tree's
    /// BVH directly (avoiding lossy slice/composition through scale-0).
    fn read_effective_blocks_at_scale(&self, chunk_key: ChunkKey, scale_exp: i8) -> Vec<BlockData> {
        // Start with virgin blocks as baseline.
        let mut blocks = self.query_virgin_blocks_at_scale(chunk_key, scale_exp);

        // Quick check: does the override tree have any data overlapping this region?
        let chunk_bounds = Aabb4i::chunk_world_bounds(chunk_key, scale_exp);
        let has_overlap = self
            .override_chunks
            .root()
            .map(|root| root.bounds.intersects(&chunk_bounds))
            .unwrap_or(false);
        if !has_overlap {
            return blocks;
        }

        // Walk the override tree's BVH for each cell to layer overrides.
        let step = step_for_scale(scale_exp);
        let cs = ChunkCoord::from_num(CHUNK_SIZE as i32);
        let origin = chunk_key.map(|k| k.saturating_mul(cs));

        for lw in 0..CHUNK_SIZE {
            for lz in 0..CHUNK_SIZE {
                for ly in 0..CHUNK_SIZE {
                    for lx in 0..CHUNK_SIZE {
                        let pos = [
                            origin[0] + step * ChunkCoord::from_num(lx as i32),
                            origin[1] + step * ChunkCoord::from_num(ly as i32),
                            origin[2] + step * ChunkCoord::from_num(lz as i32),
                            origin[3] + step * ChunkCoord::from_num(lw as i32),
                        ];
                        if let Some(block) = self.override_chunks.block_data_at(pos) {
                            let idx = lw * CHUNK_SIZE * CHUNK_SIZE * CHUNK_SIZE
                                + lz * CHUNK_SIZE * CHUNK_SIZE
                                + ly * CHUNK_SIZE
                                + lx;
                            blocks[idx] = block;
                        }
                    }
                }
            }
        }

        blocks
    }

    /// Determine the optimal chunk scale for placing a block at the given position.
    ///
    /// Tries chunk scales from `block_scale` (coarsest, fewest cells) down to
    /// `block_scale - 3` (finest, guaranteed valid since the chunk covers exactly
    /// one block AABB). Returns the coarsest scale at which all existing non-air
    /// blocks in the candidate chunk bounds are representable.
    fn determine_chunk_scale(&self, position: [ChunkCoord; 4], block_scale: i8) -> i8 {
        for candidate in (block_scale.saturating_sub(3)..=block_scale).rev() {
            let (key, _) = world_to_chunk_at_scale(
                position[0],
                position[1],
                position[2],
                position[3],
                candidate,
            );
            let chunk_bounds = Aabb4i::chunk_world_bounds(key, candidate);

            // Check that all existing blocks in this region are representable
            // at the candidate chunk scale.
            if self.all_blocks_representable_at_scale(chunk_bounds, candidate) {
                return candidate;
            }
        }
        // Fallback: finest scale always works — chunk covers exactly one block AABB.
        block_scale.saturating_sub(3)
    }

    /// Check if all non-air blocks in the given world region can be stored
    /// in a chunk at `chunk_scale`.
    ///
    /// A block with `block.scale_exp == S` is representable when:
    /// - `S >= chunk_scale` (can't store finer blocks in a coarser chunk)
    /// - `S - chunk_scale <= 3` (block can't exceed chunk size)
    fn all_blocks_representable_at_scale(&self, bounds: Aabb4i, chunk_scale: i8) -> bool {
        // Query composed (override + virgin) state for the region.
        let core = self.query_region_core(QueryVolume { bounds }, QueryDetail::Exact);
        all_blocks_in_core_representable(&core, chunk_scale)
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

    #[cfg(test)]
    pub fn dirty_save_chunk_entries(&self) -> &HashMap<ChunkKey, i8> {
        &self.dirty_save_chunks
    }

    pub fn apply_voxel_edit(
        &mut self,
        position: [ChunkCoord; 4],
        block: BlockData,
    ) -> Option<ChunkKey> {
        self.apply_voxel_edit_at_scale(position, block, 0)
    }

    /// Edit a voxel at a specific scale.
    ///
    /// Determines the optimal chunk scale, reads the effective composed state
    /// (override + virgin), stamps the block, and stores the result. The tree
    /// handles carving existing ChunkArrays and filling gaps via resampling.
    pub fn apply_voxel_edit_at_scale(
        &mut self,
        position: [ChunkCoord; 4],
        block: BlockData,
        scale_exp: i8,
    ) -> Option<ChunkKey> {
        // 1. Determine the optimal chunk scale for this placement.
        let chunk_scale = self.determine_chunk_scale(position, scale_exp);
        let (chunk_key, voxel_idx) = world_to_chunk_at_scale(
            position[0],
            position[1],
            position[2],
            position[3],
            chunk_scale,
        );
        let chunk_bounds = Aabb4i::chunk_world_bounds(chunk_key, chunk_scale);

        // 2. Ensure persisted overrides are loaded for this region.
        if let Err(e) = self.ensure_persisted_bounds_loaded(chunk_bounds) {
            eprintln!("failed to hydrate spatial region before edit: {}", e);
        }

        // 3. Read effective (override + virgin) world state for this chunk region.
        let mut blocks = self.read_effective_blocks_at_scale(chunk_key, chunk_scale);

        // 4. Compute multi-cell footprint for coarser-than-chunk placements.
        let multi_cell = if scale_exp > chunk_scale {
            let cells_per_axis = 1usize << (scale_exp - chunk_scale) as u32;
            let lx = voxel_idx % CHUNK_SIZE;
            let ly = (voxel_idx / CHUNK_SIZE) % CHUNK_SIZE;
            let lz = (voxel_idx / (CHUNK_SIZE * CHUNK_SIZE)) % CHUNK_SIZE;
            let lw = voxel_idx / (CHUNK_SIZE * CHUNK_SIZE * CHUNK_SIZE);
            Some((
                [
                    lx - (lx % cells_per_axis),
                    ly - (ly % cells_per_axis),
                    lz - (lz % cells_per_axis),
                    lw - (lw % cells_per_axis),
                ],
                cells_per_axis,
            ))
        } else {
            None
        };

        // 5. Collision check: non-air placements require all target cells to be air.
        if !block.is_air() {
            if let Some((base, cells_per_axis)) = multi_cell {
                if !all_cells_air(&blocks, base, cells_per_axis) {
                    return None;
                }
            } else if !blocks[voxel_idx].is_air() {
                return None;
            }
        }

        // 6. Stamp the block with its scale metadata.
        let block = block.at_scale(scale_exp);

        if let Some((base, cells_per_axis)) = multi_cell {
            fill_multi_cell_block(&mut blocks, base, block.clone(), cells_per_axis);
        } else {
            blocks[voxel_idx] = block;
        }

        // 7. Compare to virgin: prune if identical.
        let virgin_blocks = self.query_virgin_blocks_at_scale(chunk_key, chunk_scale);
        let should_remove = blocks_match_ignoring_scale(&blocks, &virgin_blocks);
        let payload = if should_remove {
            None
        } else {
            ResolvedChunkPayload::from_dense_blocks(&blocks).ok()
        };

        // 7. Store — the tree handles carving and gap-filling.
        let affected_bounds =
            self.override_chunks
                .set_chunk_at_scale(chunk_key, payload, chunk_scale);
        let Some(affected_bounds) = affected_bounds else {
            return None;
        };

        // 8. Dirty tracking.
        // Mark the directly-edited chunk for BVH delta updates.
        self.mark_dirty_chunk(chunk_key, chunk_scale);
        // Mark ALL chunks in the affected region dirty for save.  When the
        // edit carves an existing chunk at a different scale, `affected_bounds`
        // covers the full carved region — including fragments that hold the
        // old data re-encoded at the new scale.  Without this, those fragments
        // would not be persisted and the old data would be lost on reload.
        for (key, se) in self
            .override_chunks
            .collect_chunk_entries_in_bounds(affected_bounds)
        {
            self.dirty_save_chunks.insert(key, se);
        }
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

        let mut dirty_entries: Vec<(ChunkKey, i8)> = self
            .dirty_save_chunks
            .iter()
            .map(|(&k, &se)| (k, se))
            .collect();
        dirty_entries.sort_unstable();
        let dirty_chunk_payloads =
            dirty_entries
                .into_iter()
                .map(|(key, se)| {
                    let tree_result = self.override_chunks.chunk_payload(key);
                    debug_assert!(
                    tree_result.as_ref().map(|(_, tree_se)| *tree_se == se).unwrap_or(true),
                    "dirty_save_chunks scale {se} disagrees with tree scale {:?} for key {key:?}",
                    tree_result.as_ref().map(|(_, s)| s),
                );
                    let resolved = tree_result.map(|(p, _)| p);
                    (key, se, resolved)
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
                    players: None,
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

    pub fn chunk_at(&self, chunk_key: ChunkKey) -> Option<(ResolvedChunkPayload, i8)> {
        self.override_chunks.chunk_payload(chunk_key)
    }

    pub fn effective_chunk(
        &self,
        chunk_key: ChunkKey,
        preserve_explicit_empty_chunk: bool,
    ) -> Option<ResolvedChunkPayload> {
        if let Some((payload, _)) = self.override_chunks.chunk_payload(chunk_key) {
            return resolved_to_effective(payload, preserve_explicit_empty_chunk);
        }
        let virgin_payload = self.query_virgin_chunk_payload(chunk_key)?;
        resolved_to_effective(virgin_payload, preserve_explicit_empty_chunk)
    }
}

/// Check whether all non-air blocks in a `RegionTreeCore` can be represented
/// at the given chunk scale.
///
/// A block is representable when `block.scale_exp >= chunk_scale` and
/// `block.scale_exp - chunk_scale <= 3`.
fn all_blocks_in_core_representable(core: &RegionTreeCore, chunk_scale: i8) -> bool {
    match &core.kind {
        RegionNodeKind::Empty | RegionNodeKind::ProceduralRef(_) => true,
        RegionNodeKind::Uniform(block) => {
            if block.is_air() {
                return true;
            }
            let diff = block.scale_exp - chunk_scale;
            diff >= 0 && diff <= 3
        }
        RegionNodeKind::ChunkArray(ca) => {
            // The ChunkArray's own scale tells us the finest block scale
            // stored within. Blocks in a ChunkArray have scale_exp >= ca.scale_exp.
            // We also know block.scale_exp - ca.scale_exp <= 3 from existing
            // structural constraints. So check if the array's scale can be
            // re-encoded at chunk_scale.
            //
            // All blocks have scale_exp in [ca.scale_exp, ca.scale_exp + 3].
            // For representability: we need block.scale_exp >= chunk_scale
            // AND block.scale_exp - chunk_scale <= 3.
            // The finest block is at ca.scale_exp, so ca.scale_exp >= chunk_scale.
            // The coarsest is at ca.scale_exp + 3, so ca.scale_exp + 3 - chunk_scale <= 3
            //   ⟹ ca.scale_exp >= chunk_scale.
            // So the simple check is: ca.scale_exp >= chunk_scale.
            ca.scale_exp >= chunk_scale
        }
        RegionNodeKind::Branch(children) => children
            .iter()
            .all(|child| all_blocks_in_core_representable(child, chunk_scale)),
    }
}

/// Compare two block slices ignoring `scale_exp` on each block.
///
/// The tree's Uniform storage normalises `scale_exp` to the chunk's grid scale,
/// so override blocks read back from the tree may differ from virgin blocks only
/// in `scale_exp`. For pruning purposes this difference is irrelevant.
fn blocks_match_ignoring_scale(a: &[BlockData], b: &[BlockData]) -> bool {
    a.len() == b.len()
        && a.iter()
            .zip(b.iter())
            .all(|(x, y)| x.matches_ignoring_scale(y))
}

/// Fill multiple cells in a chunk when placing a coarser block into a finer grid.
///
/// A block at `edit_scale` placed into a chunk at `chunk_scale` (where
/// `chunk_scale < edit_scale`) fills `cells_per_axis^4` cells.
fn fill_multi_cell_block(
    blocks: &mut [BlockData],
    edit_local_base: [usize; 4],
    block: BlockData,
    cells_per_axis: usize,
) {
    for dw in 0..cells_per_axis {
        for dz in 0..cells_per_axis {
            for dy in 0..cells_per_axis {
                for dx in 0..cells_per_axis {
                    let x = edit_local_base[0] + dx;
                    let y = edit_local_base[1] + dy;
                    let z = edit_local_base[2] + dz;
                    let w = edit_local_base[3] + dw;
                    if x < CHUNK_SIZE && y < CHUNK_SIZE && z < CHUNK_SIZE && w < CHUNK_SIZE {
                        let idx = w * CHUNK_SIZE * CHUNK_SIZE * CHUNK_SIZE
                            + z * CHUNK_SIZE * CHUNK_SIZE
                            + y * CHUNK_SIZE
                            + x;
                        blocks[idx] = block.clone();
                    }
                }
            }
        }
    }
}

/// Check that all cells in a multi-cell region are air.
fn all_cells_air(blocks: &[BlockData], edit_local_base: [usize; 4], cells_per_axis: usize) -> bool {
    for dw in 0..cells_per_axis {
        for dz in 0..cells_per_axis {
            for dy in 0..cells_per_axis {
                for dx in 0..cells_per_axis {
                    let x = edit_local_base[0] + dx;
                    let y = edit_local_base[1] + dy;
                    let z = edit_local_base[2] + dz;
                    let w = edit_local_base[3] + dw;
                    if x < CHUNK_SIZE && y < CHUNK_SIZE && z < CHUNK_SIZE && w < CHUNK_SIZE {
                        let idx = w * CHUNK_SIZE * CHUNK_SIZE * CHUNK_SIZE
                            + z * CHUNK_SIZE * CHUNK_SIZE
                            + y * CHUNK_SIZE
                            + x;
                        if !blocks[idx].is_air() {
                            return false;
                        }
                    }
                }
            }
        }
    }
    true
}

fn chunk_payload_from_core(
    core: &RegionTreeCore,
    key_pos: ChunkKey,
) -> Option<ResolvedChunkPayload> {
    if !core.bounds.contains_chunk_world_min(key_pos) {
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
            .find(|child| child.bounds.contains_chunk_world_min(key_pos))
            .and_then(|child| chunk_payload_from_core(child, key_pos)),
    }
}

fn chunk_array_payload_at(chunk_array: &ChunkArrayData, key_pos: ChunkKey) -> Option<ChunkPayload> {
    if !chunk_array.bounds.contains_chunk_world_min(key_pos) {
        return None;
    }
    let dense_indices = chunk_array.decode_dense_indices().ok()?;
    let se = chunk_array.scale_exp;
    let extents = chunk_array.bounds.chunk_extents_at_scale(se)?;
    // Convert both to lattice coordinates at the chunk array's scale, then subtract.
    let (ca_lmin, _) = chunk_array.bounds.to_chunk_lattice_bounds(se);
    let lpos = [
        lattice_from_fixed(key_pos[0], se),
        lattice_from_fixed(key_pos[1], se),
        lattice_from_fixed(key_pos[2], se),
        lattice_from_fixed(key_pos[3], se),
    ];
    let local = [
        (lpos[0] - ca_lmin[0]) as usize,
        (lpos[1] - ca_lmin[1]) as usize,
        (lpos[2] - ca_lmin[2]) as usize,
        (lpos[3] - ca_lmin[3]) as usize,
    ];
    let linear = linear_cell_index(local, extents);
    let palette_idx = *dense_indices.get(linear)? as usize;
    chunk_array.chunk_palette.get(palette_idx).cloned()
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
                if is_air {
                    None
                } else {
                    Some(resolved)
                }
            }
            _ => Some(resolved),
        }
    }
}

pub type ServerWorldOverlay = PassthroughWorldOverlay<ServerWorldField>;

#[cfg(test)]
mod tests;
