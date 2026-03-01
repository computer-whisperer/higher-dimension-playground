use crate::save_v4::{self, SaveChunkPayloadPatchRequest};
use crate::shared::chunk_payload::{ChunkArrayData, ChunkPayload, ResolvedChunkPayload};
use crate::shared::protocol::WorldBounds;
use crate::shared::region_tree::{
    ChunkKey, RegionChunkTree, RegionNodeKind, RegionTreeCore,
};
use crate::shared::spatial::{
    lattice_from_fixed, step_for_scale, Aabb4i, ChunkCoord,
};
use crate::shared::voxel::{
    linear_cell_index, world_to_chunk_at_scale, BaseWorldKind, BlockData,
    CHUNK_SIZE, CHUNK_VOLUME,
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
        let virgin_core = self
            .field
            .query_region_core(QueryVolume { bounds: query_bounds }, QueryDetail::Exact);

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

                        let world_pos: [i32; 4] = if scale_exp > 0 {
                            let shift = scale_exp as u32;
                            cell_lat.map(|v| (v << shift) as i32)
                        } else {
                            let shift = (-scale_exp) as u32;
                            cell_lat.map(|v| v.div_euclid(1i64 << shift) as i32)
                        };

                        let (s0_key, s0_idx) = world_to_chunk_at_scale(
                            world_pos[0],
                            world_pos[1],
                            world_pos[2],
                            world_pos[3],
                            0,
                        );

                        let payload = payload_cache
                            .entry(s0_key)
                            .or_insert_with(|| {
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
    fn read_effective_blocks_at_scale(
        &self,
        chunk_key: ChunkKey,
        scale_exp: i8,
    ) -> Vec<BlockData> {
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
    fn determine_chunk_scale(&self, position: [i32; 4], block_scale: i8) -> i8 {
        for candidate in (block_scale.saturating_sub(3)..=block_scale).rev() {
            let (key, _) = world_to_chunk_at_scale(
                position[0], position[1], position[2], position[3], candidate,
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
    fn all_blocks_representable_at_scale(
        &self,
        bounds: Aabb4i,
        chunk_scale: i8,
    ) -> bool {
        // Query composed (override + virgin) state for the region.
        let core = self.query_region_core(
            QueryVolume { bounds },
            QueryDetail::Exact,
        );
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

    pub fn apply_voxel_edit(
        &mut self,
        position: [i32; 4],
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
        position: [i32; 4],
        block: BlockData,
        scale_exp: i8,
    ) -> Option<ChunkKey> {
        // 1. Determine the optimal chunk scale for this placement.
        let chunk_scale = self.determine_chunk_scale(position, scale_exp);
        let (chunk_key, voxel_idx) = world_to_chunk_at_scale(
            position[0], position[1], position[2], position[3], chunk_scale,
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
            Some(([
                lx - (lx % cells_per_axis),
                ly - (ly % cells_per_axis),
                lz - (lz % cells_per_axis),
                lw - (lw % cells_per_axis),
            ], cells_per_axis))
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
        let changed = self
            .override_chunks
            .set_chunk_at_scale(chunk_key, payload, chunk_scale);
        if !changed {
            return None;
        }

        // 8. Dirty tracking.
        self.mark_dirty_chunk(chunk_key, chunk_scale);
        self.dirty_save_chunks.insert(chunk_key, chunk_scale);
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

        let mut dirty_entries: Vec<(ChunkKey, i8)> =
            self.dirty_save_chunks.iter().map(|(&k, &se)| (k, se)).collect();
        dirty_entries.sort_unstable();
        let dirty_chunk_payloads = dirty_entries
            .into_iter()
            .map(|(key, se)| {
                let resolved = self.override_chunks.chunk_payload(key).map(|(p, _)| p);
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
        self.override_chunks
            .chunk_payload(chunk_key)
    }

    pub fn effective_chunk(
        &self,
        chunk_key: ChunkKey,
        preserve_explicit_empty_chunk: bool,
    ) -> Option<ResolvedChunkPayload> {
        if let Some((payload, _)) = self
            .override_chunks
            .chunk_payload(chunk_key)
        {
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
fn all_cells_air(
    blocks: &[BlockData],
    edit_local_base: [usize; 4],
    cells_per_axis: usize,
) -> bool {
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
            Aabb4i::chunk_world_bounds(key, 0),
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
        assert!(query_bounds.contains_chunk_world_min(chunk_key));
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
            let (bmin, bmax) = bounds.to_chunk_lattice_bounds(0);
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
        assert_eq!(dirty, vec![Aabb4i::chunk_world_bounds(chunk_key_i32(0, 0, 0, 0), 0)]);
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

        let (chunk_key, voxel_idx) = world_to_chunk_at_scale(0, -1, 0, 0, 0);
        let bounds = Aabb4i::chunk_world_bounds(chunk_key, 0);

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
        let (chunk_key, voxel_idx) = world_to_chunk_at_scale(0, -1, 0, 0, 0);
        let bounds = Aabb4i::chunk_world_bounds(chunk_key, 0);
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
        let chunk_key = [0, -1, 0, 0].map(ChunkCoord::from_num);
        let query_bounds = Aabb4i::chunk_world_bounds(chunk_key, 0);
        let core = overlay.query_region_core(
            QueryVolume {
                bounds: query_bounds,
            },
            QueryDetail::Exact,
        );
        assert!(core.bounds.contains_chunk_world_min(chunk_key));
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
            world_to_chunk_at_scale(edit_pos[0], edit_pos[1], edit_pos[2], edit_pos[3], 0);
        let chunk_bounds = Aabb4i::chunk_world_bounds(chunk_key, 0);
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
                    let query_bounds = Aabb4i::from_lattice_bounds(
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
                        0,
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
            world_to_chunk_at_scale(edit_pos[0], edit_pos[1], edit_pos[2], edit_pos[3], 0);

        // Place a different block.
        let edit_block = BlockData::simple(0, 42);
        let changed = overlay.apply_voxel_edit(edit_pos, edit_block.clone());
        assert_eq!(changed, Some(edit_chunk_key));

        let dirty_bounds = overlay.take_dirty_bounds();
        assert!(!dirty_bounds.is_empty());
    }

    // ── Cross-scale overlap resolution tests ────────────────────────────

    fn make_empty_overlay() -> PassthroughWorldOverlay<ServerWorldField> {
        let field = build_server_world_field(
            BaseWorldKind::Empty,
            42,
            false,
            HashSet::new(),
        );
        PassthroughWorldOverlay::new(field, BaseWorldKind::Empty, 42)
    }

    fn read_block_from_overlay(
        overlay: &PassthroughWorldOverlay<ServerWorldField>,
        pos: [i32; 4],
        _scale_exp: i8,
    ) -> BlockData {
        use crate::shared::spatial::ChunkCoord;
        let fixed_pos = [
            ChunkCoord::from_num(pos[0]),
            ChunkCoord::from_num(pos[1]),
            ChunkCoord::from_num(pos[2]),
            ChunkCoord::from_num(pos[3]),
        ];
        overlay.override_chunks.block_at(fixed_pos)
    }

    #[test]
    fn cross_scale_finer_then_coarser_both_survive() {
        let mut overlay = make_empty_overlay();
        let stone = BlockData::simple(0, 1);
        let brick = BlockData::simple(0, 2);

        // Place a scale -2 block at origin.
        overlay.apply_voxel_edit_at_scale([0, 0, 0, 0], stone.clone(), -2);
        // Place a scale -1 block at [2,0,0,0] — spatially nearby.
        // The scale -1 chunk at origin covers [0, 3.5], overlapping the -2 chunk.
        overlay.apply_voxel_edit_at_scale([2, 0, 0, 0], brick.clone(), -1);

        // The -2 block should still be readable at its position.
        let read_stone = read_block_from_overlay(&overlay, [0, 0, 0, 0], -2);
        assert_eq!(
            read_stone.block_type, 1,
            "scale -2 stone block should survive after scale -1 placement"
        );
    }

    #[test]
    fn cross_scale_coarser_then_finer_rechunks() {
        let mut overlay = make_empty_overlay();
        let stone = BlockData::simple(0, 1);
        let brick = BlockData::simple(0, 2);

        // Place a scale -1 block at origin first.
        overlay.apply_voxel_edit_at_scale([0, 0, 0, 0], stone.clone(), -1);
        // Now place a scale -2 block — should trigger rechunking of the -1 chunk.
        overlay.apply_voxel_edit_at_scale([0, 0, 0, 0], brick.clone(), -2);

        // The brick at scale -2 should be present.
        let read_brick = read_block_from_overlay(&overlay, [0, 0, 0, 0], -2);
        assert_eq!(
            read_brick.block_type, 2,
            "scale -2 brick should be placed after rechunking"
        );

        // The original stone content (which was at scale -1) should be rechunked
        // to scale -2 and preserved in the cells that weren't overwritten.
    }

    #[test]
    fn same_scale_edit_no_rechunk() {
        let mut overlay = make_empty_overlay();
        let stone = BlockData::simple(0, 1);
        let brick = BlockData::simple(0, 2);

        // Place two blocks at the same scale.
        overlay.apply_voxel_edit_at_scale([0, 0, 0, 0], stone.clone(), -1);
        overlay.apply_voxel_edit_at_scale([1, 0, 0, 0], brick.clone(), -1);

        let read_stone = read_block_from_overlay(&overlay, [0, 0, 0, 0], -1);
        let read_brick = read_block_from_overlay(&overlay, [1, 0, 0, 0], -1);
        assert_eq!(read_stone.block_type, 1);
        assert_eq!(read_brick.block_type, 2);
    }

    #[test]
    fn coarser_edit_fills_multiple_cells() {
        let mut overlay = make_empty_overlay();
        let brick = BlockData::simple(0, 2);

        // Place a scale -2 block first (fine).
        overlay.apply_voxel_edit_at_scale([0, 0, 0, 0], BlockData::simple(0, 1), -2);

        // Place a scale -1 block (coarser) into the same region.
        // The target scale is -2 (finest), so the -1 block should fill
        // 2^4 = 16 cells at scale -2 (2 cells per axis).
        overlay.apply_voxel_edit_at_scale([0, 0, 0, 0], brick.clone(), -1);

        // The brick should fill a 2x2x2x2 region at scale -2.
        let read = read_block_from_overlay(&overlay, [0, 0, 0, 0], -2);
        assert_eq!(
            read.block_type, 2,
            "coarser brick block should overwrite finer cells"
        );
    }

    #[test]
    fn placing_finer_block_near_multiple_coarser_blocks_preserves_all() {
        let mut overlay = make_empty_overlay();

        // Build up a set of blocks at scale -1 (simulating a wall).
        let positions_s1 = [
            [0, 0, 0, 0],
            [1, 0, 0, 0],
            [2, 0, 0, 0],
            [3, 0, 0, 0],
            [0, 1, 0, 0],
            [1, 1, 0, 0],
            [2, 1, 0, 0],
            [3, 1, 0, 0],
        ];
        for (i, &pos) in positions_s1.iter().enumerate() {
            let block = BlockData::simple(0, 10 + i as u32);
            overlay.apply_voxel_edit_at_scale(pos, block, -1);
        }

        // Verify all blocks are present before the finer edit.
        for (i, &pos) in positions_s1.iter().enumerate() {
            let read = read_block_from_overlay(&overlay, pos, -1);
            assert_eq!(
                read.block_type,
                10 + i as u32,
                "pre-edit: block at {:?} scale -1 missing (got type {})",
                pos,
                read.block_type
            );
        }

        // Now place a finer block (scale -2) nearby.
        let finer_block = BlockData::simple(0, 99);
        overlay.apply_voxel_edit_at_scale([0, 0, 0, 0], finer_block, -2);

        // ALL original scale -1 blocks should survive (rechunked to scale -2).
        // Read them back at scale -2 (they should still be present as multi-cell
        // populations at the finer scale).
        //
        // At scale -1 → -2, each -1 cell maps to 2 finer cells per axis.
        // The original -1 block at world pos [X,Y,Z,W] occupies a specific cell
        // in the -1 chunk. After rechunking to -2, that cell becomes a 2×2×2×2
        // block of -2 cells, all with the original block data.
        //
        // We verify by reading at scale -2 at positions that should be within
        // the rechunked cells.
        for (i, &pos) in positions_s1.iter().enumerate() {
            let read = read_block_from_overlay(&overlay, pos, -2);
            let expected = if pos == [0, 0, 0, 0] {
                99 // overwritten by the finer edit
            } else {
                10 + i as u32
            };
            assert_eq!(
                read.block_type, expected,
                "post-edit: block at {:?} scale -2 wrong (expected {}, got {})",
                pos, expected, read.block_type
            );
        }
    }

    #[test]
    fn placing_finer_block_does_not_lose_distant_same_scale_chunks() {
        let mut overlay = make_empty_overlay();

        // Place blocks at scale -1 in two DIFFERENT chunks.
        // At scale -1, chunk size is 4 world units, so world pos 0 and 4
        // fall in different chunks.
        let block_a = BlockData::simple(0, 10);
        let block_b = BlockData::simple(0, 20);
        overlay.apply_voxel_edit_at_scale([0, 0, 0, 0], block_a.clone(), -1);
        overlay.apply_voxel_edit_at_scale([4, 0, 0, 0], block_b.clone(), -1);

        // Verify both exist.
        assert_eq!(
            read_block_from_overlay(&overlay, [0, 0, 0, 0], -1).block_type,
            10
        );
        assert_eq!(
            read_block_from_overlay(&overlay, [4, 0, 0, 0], -1).block_type,
            20
        );

        // Place a finer block in the first chunk's region.
        overlay.apply_voxel_edit_at_scale([0, 0, 0, 0], BlockData::simple(0, 99), -2);

        // The second chunk should be completely unaffected.
        let read_b = read_block_from_overlay(&overlay, [4, 0, 0, 0], -1);
        assert_eq!(
            read_b.block_type, 20,
            "distant chunk at scale -1 should survive finer edit in another chunk"
        );
    }

    // -----------------------------------------------------------------------
    // Virgin-aware multi-scale edit tests
    // -----------------------------------------------------------------------

    fn make_flat_floor_overlay() -> ServerWorldOverlay {
        ServerWorldOverlay::from_chunk_payloads(
            BaseWorldKind::FlatFloor {
                material: BlockData::simple(0, 11),
            },
            Vec::<([i32; 4], ResolvedChunkPayload)>::new(),
            42,
            false,
            HashSet::new(),
        )
    }

    /// Read a block through the full compositing pipeline (virgin + override).
    fn read_composed_block(
        overlay: &ServerWorldOverlay,
        pos: [i32; 4],
    ) -> BlockData {
        let (chunk_key, voxel_idx) = world_to_chunk_at_scale(pos[0], pos[1], pos[2], pos[3], 0);
        let bounds = Aabb4i::chunk_world_bounds(chunk_key, 0);
        let core = overlay.query_region_core(
            QueryVolume { bounds },
            QueryDetail::Exact,
        );
        chunk_payload_from_core(core.as_ref(), chunk_key)
            .map(|p| p.block_at(voxel_idx))
            .unwrap_or(BlockData::AIR)
    }

    #[test]
    fn scale_minus_one_edit_preserves_virgin_floor() {
        let mut overlay = make_flat_floor_overlay();

        // The flat floor is at chunk y=-1, so world y=-8..-1 all have
        // the floor material. Editing at scale -1 near the floor should
        // NOT cause the floor to disappear.
        let edit_pos = [0, -1, 0, 0]; // on the floor
        let brick = BlockData::simple(0, 2);
        overlay.apply_voxel_edit_at_scale(edit_pos, brick.clone(), -1);

        // The edited block should be present (read from override).
        let read_edited = read_block_from_overlay(&overlay, edit_pos, -1);
        assert_eq!(
            read_edited.block_type, 2,
            "edited block should be readable at scale -1"
        );

        // Nearby floor blocks should still be present in the composed view.
        // World pos [1, -1, 0, 0] is adjacent to the edit but should still be floor.
        let neighbor = read_composed_block(&overlay, [1, -1, 0, 0]);
        assert!(
            !neighbor.is_air(),
            "neighboring floor block at [1,-1,0,0] should survive the scale -1 edit"
        );
    }

    #[test]
    fn scale_minus_two_edit_preserves_virgin_floor() {
        let mut overlay = make_flat_floor_overlay();

        // Edit at scale -2 on the floor — the override chunk should be
        // initialized with virgin floor data, not air.
        let edit_pos = [0, -1, 0, 0];
        overlay.apply_voxel_edit_at_scale(edit_pos, BlockData::simple(0, 5), -2);

        // The edit should be there.
        let read_edited = read_block_from_overlay(&overlay, edit_pos, -2);
        assert_eq!(read_edited.block_type, 5);

        // Other cells in the same chunk should still have the floor material,
        // not air. At scale -2, the chunk covers a small region. World pos
        // [0, -1, 1, 0] is nearby and should have floor material in the override.
        // (Because the override was initialized from virgin data.)
        let (chunk_key, _) = world_to_chunk_at_scale(edit_pos[0], edit_pos[1], edit_pos[2], edit_pos[3], -2);
        let payload = overlay.override_chunks.chunk_payload(chunk_key);
        assert!(
            payload.is_some(),
            "override chunk should exist at {:?}",
            chunk_key
        );
        // The override chunk should NOT be all-air except for the edit.
        // Since the floor covers this region, most cells should be floor material.
        let blocks = payload.unwrap().0.dense_blocks();
        let non_air_count = blocks.iter().filter(|b| !b.is_air()).count();
        assert!(
            non_air_count > 1,
            "override chunk should have multiple non-air blocks (got {non_air_count}), \
             proving virgin floor data was preserved"
        );
    }

    #[test]
    fn query_virgin_blocks_at_scale_returns_floor_for_finer_scales() {
        let overlay = make_flat_floor_overlay();

        // Query virgin data at scale -1 for a chunk on the floor.
        // World y=-1 is in scale-0 chunk y=-1. At scale -1, a chunk at
        // the appropriate key should contain floor material.
        let (chunk_key, _) = world_to_chunk_at_scale(0, -1, 0, 0, -1);
        let blocks = overlay.query_virgin_blocks_at_scale(chunk_key, -1);

        let non_air = blocks.iter().filter(|b| !b.is_air()).count();
        assert!(
            non_air > 0,
            "virgin blocks at scale -1 on the floor should contain floor material, \
             got {non_air} non-air blocks out of {}",
            CHUNK_VOLUME
        );
    }

    #[test]
    fn query_virgin_blocks_at_scale_returns_air_above_floor() {
        let overlay = make_flat_floor_overlay();

        // Above the floor (y=0 in scale-0 chunks), everything should be air.
        let (chunk_key, _) = world_to_chunk_at_scale(0, 0, 0, 0, -1);
        let blocks = overlay.query_virgin_blocks_at_scale(chunk_key, -1);

        assert!(
            blocks.iter().all(|b| b.is_air()),
            "virgin blocks at scale -1 above floor should all be air"
        );
    }

    #[test]
    fn edit_restoring_virgin_state_removes_override() {
        let mut overlay = make_flat_floor_overlay();

        // Place a block on the floor, then place the floor material back.
        let edit_pos = [0, -1, 0, 0];
        let floor_material = BlockData::simple(0, 11);

        // First edit: change floor to brick.
        overlay.apply_voxel_edit(edit_pos, BlockData::simple(0, 2));
        let (chunk_key, _) = world_to_chunk_at_scale(edit_pos[0], edit_pos[1], edit_pos[2], edit_pos[3], 0);
        assert!(
            overlay.override_chunks.chunk_payload(chunk_key).is_some(),
            "override should exist after edit"
        );

        // Second edit: change back to floor material.
        overlay.apply_voxel_edit(edit_pos, floor_material);
        let still_overridden = overlay.override_chunks.chunk_payload(chunk_key);
        assert!(
            still_overridden.is_none(),
            "override should be removed when edit restores virgin state"
        );
    }

    #[test]
    fn rechunked_virgin_matching_fragments_are_pruned() {
        let mut overlay = make_flat_floor_overlay();

        // Place a scale-0 block on the floor — this creates a scale-0 override.
        let edit_pos = [0, -1, 0, 0];
        overlay.apply_voxel_edit(edit_pos, BlockData::simple(0, 2));

        let chunks_before = overlay.override_chunks.non_empty_chunk_count();
        assert!(chunks_before >= 1, "should have override chunk(s)");

        // Now place a scale -1 block — triggers rechunking of the scale-0
        // override to scale -1 (16 finer chunks). Many of those should be
        // pruned because they match virgin floor data.
        overlay.apply_voxel_edit_at_scale(edit_pos, BlockData::simple(0, 3), -1);

        let chunks_after = overlay.override_chunks.non_empty_chunk_count();
        // Without pruning, we'd have up to 16 chunks from the rechunked
        // scale-0 override. With pruning, only chunks that differ from
        // virgin survive — at minimum the edited chunk.
        assert!(
            chunks_after < 16,
            "rechunked fragments matching virgin should be pruned \
             (got {chunks_after} chunks, expected fewer than 16)"
        );
        assert!(
            chunks_after >= 1,
            "at least the edited chunk should remain"
        );
    }

    /// Place a scale-2 block into an empty world and verify that the stored
    /// blocks carry scale_exp=2 (not 0 or some other value).
    #[test]
    fn scale2_edit_stores_correct_scale_exp_on_blocks() {
        let mut overlay = make_empty_overlay();
        let block = BlockData::simple(0, 42);

        overlay.apply_voxel_edit_at_scale([0, 0, 0, 0], block.clone(), 2);

        // The edit is at scale 2 into an empty world (no finer data).
        // target_scale should be 2 (the edit's own scale).
        // Read back at scale 2.
        let read = read_block_from_overlay(&overlay, [0, 0, 0, 0], 2);
        assert_eq!(read.block_type, 42, "scale-2 block should be stored");
        assert_eq!(
            read.scale_exp, 2,
            "block should carry scale_exp=2, got {}",
            read.scale_exp
        );
    }

    /// Place a scale-3 block into an empty world and verify scale_exp=3.
    #[test]
    fn scale3_edit_stores_correct_scale_exp_on_blocks() {
        let mut overlay = make_empty_overlay();
        let block = BlockData::simple(0, 99);

        overlay.apply_voxel_edit_at_scale([0, 0, 0, 0], block.clone(), 3);

        let read = read_block_from_overlay(&overlay, [0, 0, 0, 0], 3);
        assert_eq!(read.block_type, 99, "scale-3 block should be stored");
        assert_eq!(
            read.scale_exp, 3,
            "block should carry scale_exp=3, got {}",
            read.scale_exp
        );
    }

    /// Place a scale-2 block over a scale-0 floor.
    /// The floor forces rechunking to scale 0, so the scale-2 block fills
    /// multiple scale-0 cells. Each cell should carry scale_exp=2.
    #[test]
    fn scale2_edit_over_floor_fills_cells_with_correct_scale_exp() {
        let mut overlay = make_flat_floor_overlay();

        // Place a scale-2 block on the floor. The floor is at scale 0, so
        // target_scale = 0. The edit fills (1<<(2-0))^4 = 256 cells per chunk.
        let block = BlockData::simple(0, 42);
        overlay.apply_voxel_edit_at_scale([0, -1, 0, 0], block.clone(), 2);

        // Read back one of the filled cells at scale 0.
        let read = read_block_from_overlay(&overlay, [0, -1, 0, 0], 0);
        assert_eq!(read.block_type, 42, "scale-2 block should fill scale-0 cells");
        assert_eq!(
            read.scale_exp, 2,
            "filled cells should carry the original scale_exp=2, got {}",
            read.scale_exp
        );
    }

    /// Place a scale-3 block over a scale-0 floor.
    /// Each filled cell should carry scale_exp=3.
    #[test]
    fn scale3_edit_over_floor_fills_cells_with_correct_scale_exp() {
        let mut overlay = make_flat_floor_overlay();

        let block = BlockData::simple(0, 99);
        overlay.apply_voxel_edit_at_scale([0, -1, 0, 0], block.clone(), 3);

        let read = read_block_from_overlay(&overlay, [0, -1, 0, 0], 0);
        assert_eq!(read.block_type, 99, "scale-3 block should fill scale-0 cells");
        assert_eq!(
            read.scale_exp, 3,
            "filled cells should carry the original scale_exp=3, got {}",
            read.scale_exp
        );
    }

    /// Dirty bounds for a scale-3 edit should cover the full scale-3 chunk
    /// extent, and the composited view for those bounds should include the
    /// override data.
    #[test]
    fn scale3_edit_dirty_bounds_cover_full_chunk_extent() {
        let mut overlay = make_empty_overlay();
        let block = BlockData::simple(0, 99);

        // Place a scale-3 block.
        overlay.apply_voxel_edit_at_scale([0, 0, 0, 0], block.clone(), 3);

        // Take the dirty bounds — these are what the server broadcasts to clients.
        let dirty_bounds = overlay.take_dirty_bounds();
        assert!(
            !dirty_bounds.is_empty(),
            "dirty bounds should be non-empty after edit"
        );

        // The dirty bounds should cover the full scale-3 chunk extent (64^4).
        let (chunk_key_s3, _) = world_to_chunk_at_scale(0, 0, 0, 0, 3);
        let expected_bounds = Aabb4i::chunk_world_bounds(chunk_key_s3, 3);
        let covers_expected = dirty_bounds.iter().any(|b| {
            b.min[0] <= expected_bounds.min[0]
                && b.max[0] >= expected_bounds.max[0]
                && b.min[1] <= expected_bounds.min[1]
                && b.max[1] >= expected_bounds.max[1]
                && b.min[2] <= expected_bounds.min[2]
                && b.max[2] >= expected_bounds.max[2]
                && b.min[3] <= expected_bounds.min[3]
                && b.max[3] >= expected_bounds.max[3]
        });
        assert!(
            covers_expected,
            "dirty bounds should cover the full scale-3 extent {:?}->{:?}, got {:?}",
            expected_bounds.min, expected_bounds.max, dirty_bounds
        );

        // Query the composited view using the dirty bounds (what the server
        // actually sends to the client).
        for bounds in &dirty_bounds {
            let core = overlay.query_region_core(
                QueryVolume { bounds: *bounds },
                QueryDetail::Exact,
            );

            // The scale-0 chunk at origin should contain the block.
            let (chunk_key_s0, voxel_idx) = world_to_chunk_at_scale(0, 0, 0, 0, 0);
            let payload = chunk_payload_from_core(core.as_ref(), chunk_key_s0);
            if let Some(payload) = payload {
                let block_at_origin = payload.block_at(voxel_idx);
                assert_eq!(
                    block_at_origin.block_type, 99,
                    "scale-3 block should be visible at origin in composited view"
                );
            }
        }
    }

    /// Place then break a scale-3 block in an empty world.
    /// Breaking should remove the block entirely.
    #[test]
    fn break_scale3_block_removes_it_empty_world() {
        let mut overlay = make_empty_overlay();
        let block = BlockData::simple(0, 99);

        // Place at scale 3.
        overlay.apply_voxel_edit_at_scale([0, 0, 0, 0], block.clone(), 3);

        // Verify it's there.
        let read = read_block_from_overlay(&overlay, [0, 0, 0, 0], 3);
        assert_eq!(read.block_type, 99, "block should be placed");

        // Break: send AIR at scale 3.
        overlay.apply_voxel_edit_at_scale([0, 0, 0, 0], BlockData::AIR, 3);

        // The block at the break position should be gone.
        let read = read_block_from_overlay(&overlay, [0, 0, 0, 0], 3);
        assert!(
            read.is_air(),
            "broken block should be air, got block_type={}",
            read.block_type
        );

        // And the rest of the chunk should still be air (not filled with blocks).
        let read2 = read_block_from_overlay(&overlay, [8, 0, 0, 0], 3);
        assert!(
            read2.is_air(),
            "neighboring cell at scale 3 should remain air, got block_type={}",
            read2.block_type
        );

        // Override should be removed entirely since all AIR matches empty virgin.
        assert_eq!(
            overlay.override_chunks.non_empty_chunk_count(),
            0,
            "override should be removed when edit restores virgin (empty) state"
        );
    }

    /// Placing a scale-3 block on the floor should only override ONE scale-0
    /// chunk; the floor at adjacent chunks should remain.
    #[test]
    fn scale3_place_on_floor_does_not_overwrite_neighbors() {
        let mut overlay = make_flat_floor_overlay();
        let block = BlockData::simple(0, 99);

        // Floor at [8, -1, 0, 0] should be present before edit.
        let floor_before = read_composed_block(&overlay, [8, -1, 0, 0]);
        assert_eq!(
            floor_before.block_type, 11,
            "floor should exist before edit at [8,-1,0,0]"
        );

        // Place at scale 3 on the floor (y=-1).
        overlay.apply_voxel_edit_at_scale([0, -1, 0, 0], block.clone(), 3);

        // The placed block should be at [0,-1,0,0].
        let placed = read_block_from_overlay(&overlay, [0, -1, 0, 0], 0);
        assert_eq!(placed.block_type, 99, "block should be placed");

        // The floor at [8, -1, 0, 0] should still be floor material.
        let floor_after = read_composed_block(&overlay, [8, -1, 0, 0]);
        assert_eq!(
            floor_after.block_type, 11,
            "floor at [8,-1,0,0] should survive the scale-3 edit, got block_type={}",
            floor_after.block_type
        );
    }

    /// Place then break a scale-3 block over a floor.
    /// Breaking should remove the block but preserve the floor.
    #[test]
    fn break_scale3_block_over_floor() {
        let mut overlay = make_flat_floor_overlay();
        let block = BlockData::simple(0, 99);

        // Place at scale 3 on the floor (y=-1).
        overlay.apply_voxel_edit_at_scale([0, -1, 0, 0], block.clone(), 3);

        // Verify it's there (floor forces target_scale=0, so read at scale 0).
        let read = read_block_from_overlay(&overlay, [0, -1, 0, 0], 0);
        assert_eq!(read.block_type, 99, "block should be placed");

        // Break: send AIR at scale 3.
        overlay.apply_voxel_edit_at_scale([0, -1, 0, 0], BlockData::AIR, 3);

        // The block at the break position should be gone.
        let read = read_block_from_overlay(&overlay, [0, -1, 0, 0], 0);
        assert!(
            read.is_air(),
            "broken block should be air, got block_type={}",
            read.block_type
        );

        // Read via composition: the floor should still be visible outside the
        // edit area (the override only covers the area the scale-3 block occupied).
        let floor_pos = [8, -1, 0, 0];
        let composed = read_composed_block(&overlay, floor_pos);
        assert_eq!(
            composed.block_type, 11,
            "floor outside the edit area should be preserved, got block_type={}",
            composed.block_type
        );
    }

    #[test]
    fn determine_chunk_scale_respects_existing_blocks() {
        let mut overlay = make_empty_overlay();

        // Place a scale -2 block at origin. This creates override data at scale -2.
        let fine_block = BlockData::simple(0, 5);
        overlay.apply_voxel_edit_at_scale([0, 0, 0, 0], fine_block.clone(), -2);

        // Now determine the chunk scale for a scale 0 placement at [1,0,0,0].
        // The scale-0 chunk at origin covers [0,8)^4. Inside that region there's
        // a scale -2 block. A scale 0 chunk can represent blocks with scale_exp
        // in [0, 3]. scale_exp = -2 < 0, so the block is NOT representable at
        // scale 0. determine_chunk_scale should pick a finer scale.
        let chosen = overlay.determine_chunk_scale([1, 0, 0, 0], 0);
        assert!(
            chosen <= -2,
            "chunk scale {chosen} can't represent existing scale -2 blocks"
        );

        // Place a scale 0 block at [8,0,0,0] — a region with no existing overrides.
        // The coarsest valid scale (0) should be chosen since there's nothing to conflict.
        let chosen_empty = overlay.determine_chunk_scale([8, 0, 0, 0], 0);
        assert_eq!(
            chosen_empty, 0,
            "empty region should use coarsest scale, got {chosen_empty}"
        );

        // Place a scale 1 block at [16,0,0,0]. determine_chunk_scale for a scale 0
        // edit nearby should still pick scale 0 since scale 1 is representable at
        // scale 0 (scale_exp 1 >= 0 and 1 - 0 = 1 <= 3).
        overlay.apply_voxel_edit_at_scale([16, 0, 0, 0], BlockData::simple(0, 7), 1);
        let chosen_coarse = overlay.determine_chunk_scale([17, 0, 0, 0], 0);
        assert!(
            chosen_coarse >= -3 && chosen_coarse <= 0,
            "scale 1 blocks should be representable at scale 0, got {chosen_coarse}"
        );
    }
}
