use crate::content_registry::ContentRegistry;
use crate::shared::chunk_payload::{ChunkArrayData, ChunkPayload, ResolvedChunkPayload};
use crate::shared::region_tree::{ChunkKey, RegionChunkTree, RegionNodeKind, RegionTreeCore};
use crate::shared::spatial::{
    fixed_from_lattice, step_for_scale, Aabb4i, ChunkCoord,
};
use crate::shared::voxel::{linear_cell_index, BlockData, CHUNK_SIZE, CHUNK_VOLUME};
use std::collections::{HashMap, HashSet};

/// Per-instance entry for a ticking block tracked by the cache.
#[derive(Clone, Debug)]
pub(super) struct TickingBlockEntry {
    pub namespace: u32,
    pub block_type: u32,
    pub last_tick_ms: u64,
}

/// Server-side world cache. Provides chunk data for mob collision/pathfinding
/// and maintains an index of ticking block instances.
///
/// Updated through `absorb_subtree`, which is the same path used to push
/// world data to clients. All world mutations flow through the overlay's
/// dirty-bounds mechanism, so this cache stays consistent without needing
/// separate notification paths for edits, explosions, etc.
pub(super) struct ServerWorldCache {
    chunks: RegionChunkTree,
    ticking_blocks: HashMap<[ChunkCoord; 4], TickingBlockEntry>,
    ticking_types: HashSet<(u32, u32)>,
}

impl ServerWorldCache {
    pub(super) fn new(registry: &ContentRegistry) -> Self {
        let ticking_types: HashSet<(u32, u32)> = registry
            .ticking_block_types()
            .into_iter()
            .map(|(ns, bt, _cfg)| (ns, bt))
            .collect();
        if !ticking_types.is_empty() {
            eprintln!(
                "server world cache: tracking {} ticking block type(s)",
                ticking_types.len()
            );
        }
        Self {
            chunks: RegionChunkTree::new(),
            ticking_blocks: HashMap::new(),
            ticking_types,
        }
    }

    // -----------------------------------------------------------------------
    // Chunk data access (mob collision / pathfinding)
    // -----------------------------------------------------------------------

    pub(super) fn chunk_payload(&self, chunk_key: ChunkKey) -> Option<(ResolvedChunkPayload, i8)> {
        self.chunks.chunk_payload(chunk_key)
    }

    // -----------------------------------------------------------------------
    // Ticking block access (block tick loop)
    // -----------------------------------------------------------------------

    pub(super) fn ticking_blocks(&self) -> &HashMap<[ChunkCoord; 4], TickingBlockEntry> {
        &self.ticking_blocks
    }

    pub(super) fn update_tick_time(&mut self, pos: &[ChunkCoord; 4], now_ms: u64) {
        if let Some(entry) = self.ticking_blocks.get_mut(pos) {
            entry.last_tick_ms = now_ms;
        }
    }

    // -----------------------------------------------------------------------
    // Absorption — single path for world data entering the cache
    // -----------------------------------------------------------------------

    /// Absorb a subtree patch. The `bounds` are the authoritative region —
    /// any previously tracked ticking blocks within `bounds` are invalidated,
    /// then the incoming core is scanned for new ones.
    pub(super) fn absorb_subtree(&mut self, bounds: Aabb4i, subtree: &RegionTreeCore) {
        if !bounds.is_valid() {
            return;
        }

        // Preserve last_tick_ms for entries about to be invalidated, so that
        // re-scanned positions retain their tick timing across absorption.
        let mut preserved_tick_times: HashMap<[ChunkCoord; 4], u64> = HashMap::new();
        if !self.ticking_types.is_empty() && !self.ticking_blocks.is_empty() {
            self.ticking_blocks.retain(|pos, entry| {
                if bounds.contains_point(*pos) {
                    if entry.last_tick_ms > 0 {
                        preserved_tick_times.insert(*pos, entry.last_tick_ms);
                    }
                    false
                } else {
                    true
                }
            });
        }

        // Splice chunk data.
        let _ = self.chunks.splice_core_in_bounds(bounds, subtree);

        // Scan incoming core for ticking blocks.
        if !self.ticking_types.is_empty() {
            self.scan_core_for_ticking_blocks(subtree);

            // Restore preserved tick times for positions that reappeared.
            if !preserved_tick_times.is_empty() {
                for (pos, tick_ms) in preserved_tick_times {
                    if let Some(entry) = self.ticking_blocks.get_mut(&pos) {
                        entry.last_tick_ms = tick_ms;
                    }
                }
            }
        }
    }

    // -----------------------------------------------------------------------
    // Eviction
    // -----------------------------------------------------------------------

    pub(super) fn evict_outside_bounds(
        &mut self,
        keep_bounds: Aabb4i,
        max_subtree_drops: usize,
    ) -> Option<Aabb4i> {
        let evicted = self
            .chunks
            .lazy_drop_outside_bounds(keep_bounds, max_subtree_drops);

        if evicted.is_some() && !self.ticking_blocks.is_empty() {
            self.ticking_blocks
                .retain(|pos, _| keep_bounds.contains_point(*pos));
        }

        evicted
    }

    // -----------------------------------------------------------------------
    // Tree scanning
    // -----------------------------------------------------------------------

    /// Walk the incoming core recursively, checking for ticking block types.
    /// Short-circuits at each level: Uniform regions are a single check,
    /// ChunkArrays check the block_palette before iterating voxels.
    fn scan_core_for_ticking_blocks(&mut self, core: &RegionTreeCore) {
        self.visit_core(core);
    }

    fn visit_core(&mut self, core: &RegionTreeCore) {
        match &core.kind {
            RegionNodeKind::Empty | RegionNodeKind::ProceduralRef(_) => {}

            RegionNodeKind::Uniform(block) => {
                if self.is_ticking_type(block) {
                    self.register_uniform_ticking_region(block, core.bounds);
                }
            }

            RegionNodeKind::ChunkArray(chunk_array) => {
                // Fast palette check: skip entire array if no ticking types.
                let has_ticking = chunk_array
                    .block_palette
                    .iter()
                    .any(|b| self.is_ticking_type(b));
                if has_ticking {
                    self.scan_chunk_array(chunk_array);
                }
            }

            RegionNodeKind::Branch(children) => {
                for child in children {
                    self.visit_core(child);
                }
            }
        }
    }

    fn is_ticking_type(&self, block: &BlockData) -> bool {
        !block.is_air() && self.ticking_types.contains(&(block.namespace, block.block_type))
    }

    /// A uniform region of a ticking block. Enumerate all voxel positions.
    /// This is extremely rare (a huge uniform region of spawners) but correct.
    fn register_uniform_ticking_region(&mut self, block: &BlockData, bounds: Aabb4i) {
        let step = ChunkCoord::from_num(1); // scale 0 voxels
        let mut w = bounds.min[3];
        while w < bounds.max[3] {
            let mut z = bounds.min[2];
            while z < bounds.max[2] {
                let mut y = bounds.min[1];
                while y < bounds.max[1] {
                    let mut x = bounds.min[0];
                    while x < bounds.max[0] {
                        let pos = [x, y, z, w];
                        self.ticking_blocks.entry(pos).or_insert(TickingBlockEntry {
                            namespace: block.namespace,
                            block_type: block.block_type,
                            last_tick_ms: 0,
                        });
                        x += step;
                    }
                    y += step;
                }
                z += step;
            }
            w += step;
        }
    }

    /// Scan a ChunkArray for ticking blocks. Iterates the array's chunks,
    /// and for each non-empty chunk, checks individual voxels.
    fn scan_chunk_array(&mut self, chunk_array: &ChunkArrayData) {
        let se = chunk_array.scale_exp;
        let Ok(indices) = chunk_array.decode_dense_indices() else {
            return;
        };
        let Some(extents) = chunk_array
            .bounds
            .chunk_extents_at_scale(se)
        else {
            return;
        };
        let (ca_lmin, ca_lmax) = chunk_array.bounds.to_chunk_lattice_bounds(se);
        let step = step_for_scale(se);
        let cs = ChunkCoord::from_num(CHUNK_SIZE as i32);

        for lw in ca_lmin[3]..=ca_lmax[3] {
            for lz in ca_lmin[2]..=ca_lmax[2] {
                for ly in ca_lmin[1]..=ca_lmax[1] {
                    for lx in ca_lmin[0]..=ca_lmax[0] {
                        let local = [
                            (lx - ca_lmin[0]) as usize,
                            (ly - ca_lmin[1]) as usize,
                            (lz - ca_lmin[2]) as usize,
                            (lw - ca_lmin[3]) as usize,
                        ];
                        let linear = linear_cell_index(local, extents);
                        let Some(&palette_idx) = indices.get(linear) else {
                            continue;
                        };
                        let Some(chunk_payload) =
                            chunk_array.chunk_palette.get(palette_idx as usize)
                        else {
                            continue;
                        };

                        // Build a ResolvedChunkPayload to use block_at().
                        let resolved = ResolvedChunkPayload {
                            payload: chunk_payload.clone(),
                            block_palette: chunk_array.block_palette.clone(),
                        };

                        // Quick check: does this chunk's payload reference
                        // any ticking block palette indices?
                        if !self.chunk_payload_may_contain_ticking(&resolved) {
                            continue;
                        }

                        // Compute chunk world origin.
                        let chunk_origin = [
                            fixed_from_lattice(lx, se) * cs,
                            fixed_from_lattice(ly, se) * cs,
                            fixed_from_lattice(lz, se) * cs,
                            fixed_from_lattice(lw, se) * cs,
                        ];

                        self.scan_chunk_voxels(&resolved, chunk_origin, step);
                    }
                }
            }
        }
    }

    /// Check if a resolved chunk payload could contain ticking blocks.
    /// For Uniform payloads, checks the single block.
    /// For Dense/PalettePacked, conservatively returns true if the block
    /// palette has any ticking types (already checked at ChunkArray level).
    fn chunk_payload_may_contain_ticking(&self, resolved: &ResolvedChunkPayload) -> bool {
        match &resolved.payload {
            ChunkPayload::Empty | ChunkPayload::Virgin => false,
            ChunkPayload::Uniform(idx) => {
                if let Some(block) = resolved.block_palette.get(*idx as usize) {
                    self.is_ticking_type(block)
                } else {
                    false
                }
            }
            // For dense payloads, we already know the block_palette has a
            // ticking type (checked at ChunkArray level), so return true.
            _ => true,
        }
    }

    /// Iterate voxels in a single chunk and register ticking blocks.
    fn scan_chunk_voxels(
        &mut self,
        resolved: &ResolvedChunkPayload,
        chunk_origin: [ChunkCoord; 4],
        step: ChunkCoord,
    ) {
        for idx in 0..CHUNK_VOLUME {
            let block = resolved.block_at(idx);
            if self.is_ticking_type(&block) {
                let lx = (idx % CHUNK_SIZE) as i32;
                let ly = ((idx / CHUNK_SIZE) % CHUNK_SIZE) as i32;
                let lz = ((idx / (CHUNK_SIZE * CHUNK_SIZE)) % CHUNK_SIZE) as i32;
                let lw = (idx / (CHUNK_SIZE * CHUNK_SIZE * CHUNK_SIZE)) as i32;
                let pos = [
                    chunk_origin[0] + ChunkCoord::from_num(lx) * step,
                    chunk_origin[1] + ChunkCoord::from_num(ly) * step,
                    chunk_origin[2] + ChunkCoord::from_num(lz) * step,
                    chunk_origin[3] + ChunkCoord::from_num(lw) * step,
                ];
                self.ticking_blocks.entry(pos).or_insert(TickingBlockEntry {
                    namespace: block.namespace,
                    block_type: block.block_type,
                    last_tick_ms: 0,
                });
            }
        }
    }
}
