use super::{QueryDetail, QueryVolume, WorldField};
use crate::content_registry::ContentRegistry;
use crate::server::procgen;
use crate::shared::chunk_payload::{ChunkArrayData, ChunkPayload};
#[cfg(test)]
use crate::shared::chunk_payload::ResolvedChunkPayload;
use crate::shared::region_tree::{
    RegionChunkTree, RegionNodeKind, RegionTreeCore,
};
use crate::shared::spatial::Aabb4i;
use crate::shared::voxel::{BaseWorldKind, BlockData, CHUNK_VOLUME};
#[cfg(test)]
use crate::shared::voxel::ChunkPos;
use std::collections::{HashMap, HashSet};
use std::sync::Arc;

type DenseChunk = [u16; CHUNK_VOLUME];

// Poisson-disk grid cell sizes.
const POISSON_CELL_XZW: i32 = 48;
const POISSON_CELL_Y: i32 = 12;
// Anisotropic scaling: Y distances are multiplied by this before comparison.
const POISSON_Y_SCALE: i64 = 4; // = POISSON_CELL_XZW / POISSON_CELL_Y
// Minimum squared distance in scaled space between accepted candidates.
const POISSON_MIN_DIST_SQ: i64 = 2304; // 48 * 48

const PLATFORM_THICKNESS_CHUNKS: i32 = 2;
const PLATFORM_HALF_EXTENT_MIN_CHUNKS: i32 = 10;
const PLATFORM_HALF_EXTENT_MAX_CHUNKS: i32 = 18;
const ORIGIN_ANCHOR_HALF_EXTENT_CHUNKS: i32 = 18;
const ORIGIN_ANCHOR_POS: [i32; 4] = [0, 0, 0, 0];

const PLATFORM_HASH_SALT_EXTENT_X: u64 = 0x58d7_6c29_2a0b_4cd1;
const PLATFORM_HASH_SALT_EXTENT_Z: u64 = 0x7ef8_18a3_02e5_2b3f;
const PLATFORM_HASH_SALT_EXTENT_W: u64 = 0x4cc2_743e_91ad_0627;
const SALT_CANDIDATE_X: u64 = 0x0a1e_d952_75c4_6619;
const SALT_CANDIDATE_Y: u64 = 0x8f63_d4b2_3c14_20e3;
const SALT_CANDIDATE_Z: u64 = 0x3ce7_b890_5d61_489f;
const SALT_CANDIDATE_W: u64 = 0xd4f7_3a61_e28b_5c09;
const SALT_PRIORITY: u64 = 0x91f4_2b7e_87f1_3a55;
const PLATFORM_HASH_SALT_PROCGEN_SEED: u64 = 0x2b8a_4e19_9df3_512c;
const PLATFORM_HASH_SALT_PROCGEN_FRAME: u64 = 0x7a63_1fbc_3d71_2a94;
const PLATFORM_PROCGEN_FRAME_BASE_OFFSET_CHUNKS: i32 = 4096;

#[derive(Clone, Copy, Debug)]
struct PlatformInstance {
    level: i32,
    cell_x: i32,
    cell_z: i32,
    cell_w: i32,
    bounds: Aabb4i,
}

pub struct MassivePlatformsWorldGenerator {
    platform_voxel: Option<BlockData>,
    world_seed: u64,
    procgen_structures: bool,
    blocked_cells: HashSet<procgen::StructureCell>,
    content_registry: Arc<ContentRegistry>,
}

impl std::fmt::Debug for MassivePlatformsWorldGenerator {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("MassivePlatformsWorldGenerator")
            .field("platform_voxel", &self.platform_voxel)
            .field("world_seed", &self.world_seed)
            .field("procgen_structures", &self.procgen_structures)
            .field("blocked_cells", &self.blocked_cells)
            .finish()
    }
}

impl MassivePlatformsWorldGenerator {
    pub fn from_chunk_payloads(
        base_kind: BaseWorldKind,
        _chunk_payloads: impl IntoIterator<Item = ([i32; 4], ChunkPayload)>,
        world_seed: u64,
        procgen_structures: bool,
        blocked_cells: HashSet<procgen::StructureCell>,
        content_registry: Arc<ContentRegistry>,
    ) -> Self {
        Self {
            platform_voxel: platform_voxel_from_base_kind(base_kind),
            world_seed,
            procgen_structures,
            blocked_cells,
            content_registry,
        }
    }

    fn procgen_keepout_cells(&self) -> Option<&HashSet<procgen::StructureCell>> {
        if self.blocked_cells.is_empty() {
            None
        } else {
            Some(&self.blocked_cells)
        }
    }

    fn sample_platform_bounds(
        &self,
        cy: i32,
        cx: i32,
        cz: i32,
        cw: i32,
    ) -> Option<Aabb4i> {
        let material = self.platform_voxel.as_ref()?;
        if material.is_air() {
            return None;
        }

        if is_origin_cell(cx, cy, cz, cw) {
            let h = ORIGIN_ANCHOR_HALF_EXTENT_CHUNKS;
            let min_y = ORIGIN_ANCHOR_POS[1] - PLATFORM_THICKNESS_CHUNKS + 1;
            let max_y = ORIGIN_ANCHOR_POS[1];
            return Some(Aabb4i::new(
                [-h, min_y, -h, -h],
                [h - 1, max_y, h - 1, h - 1],
            ));
        }

        if !is_candidate_accepted(self.world_seed, cx, cy, cz, cw) {
            return None;
        }

        let pos = candidate_position_xyzw(self.world_seed, cx, cy, cz, cw);

        let extent_span =
            (PLATFORM_HALF_EXTENT_MAX_CHUNKS - PLATFORM_HALF_EXTENT_MIN_CHUNKS + 1).max(1) as u64;
        let half_x = PLATFORM_HALF_EXTENT_MIN_CHUNKS
            + (hash_platform_cell(
                self.world_seed,
                cy,
                cx,
                cz,
                cw,
                PLATFORM_HASH_SALT_EXTENT_X,
            ) % extent_span) as i32;
        let half_z = PLATFORM_HALF_EXTENT_MIN_CHUNKS
            + (hash_platform_cell(
                self.world_seed,
                cy,
                cx,
                cz,
                cw,
                PLATFORM_HASH_SALT_EXTENT_Z,
            ) % extent_span) as i32;
        let half_w = PLATFORM_HALF_EXTENT_MIN_CHUNKS
            + (hash_platform_cell(
                self.world_seed,
                cy,
                cx,
                cz,
                cw,
                PLATFORM_HASH_SALT_EXTENT_W,
            ) % extent_span) as i32;

        let min_y = pos[1] - PLATFORM_THICKNESS_CHUNKS + 1;
        let max_y = pos[1];

        let platform_bounds = Aabb4i::new(
            [
                pos[0] - half_x,
                min_y,
                pos[2] - half_z,
                pos[3] - half_w,
            ],
            [
                pos[0] + half_x - 1,
                max_y,
                pos[2] + half_z - 1,
                pos[3] + half_w - 1,
            ],
        );
        platform_bounds.is_valid().then_some(platform_bounds)
    }

    fn for_each_platform_bounds_intersecting_query<F>(&self, bounds: Aabb4i, mut visitor: F)
    where
        F: FnMut(Aabb4i),
    {
        if !bounds.is_valid() || self.platform_voxel.is_none() {
            return;
        }
        // Candidate Y = platform surface (bounds.max[1]).
        // Platform spans [surface - THICKNESS + 1, surface].
        // To intersect query [qmin_y, qmax_y]:
        //   surface >= qmin_y AND surface - THICKNESS + 1 <= qmax_y
        //   surface in [qmin_y, qmax_y + THICKNESS - 1]
        let min_cy = bounds.min[1].div_euclid(POISSON_CELL_Y);
        let max_cy = (bounds.max[1] + PLATFORM_THICKNESS_CHUNKS - 1).div_euclid(POISSON_CELL_Y);
        self.for_each_platform_instance_in_bounds(bounds, min_cy, max_cy, |platform| {
            let Some(_intersection) = intersect_bounds(bounds, platform.bounds) else {
                return;
            };
            visitor(platform.bounds);
        });
    }

    fn for_each_platform_instance_in_bounds<F>(
        &self,
        bounds: Aabb4i,
        min_cy: i32,
        max_cy: i32,
        mut visitor: F,
    ) where
        F: FnMut(PlatformInstance),
    {
        if !bounds.is_valid() || self.platform_voxel.is_none() || min_cy > max_cy {
            return;
        }

        let xzw_margin = POISSON_CELL_XZW - 1 + PLATFORM_HALF_EXTENT_MAX_CHUNKS;
        let min_cx = (bounds.min[0] - xzw_margin).div_euclid(POISSON_CELL_XZW);
        let max_cx = (bounds.max[0] + xzw_margin).div_euclid(POISSON_CELL_XZW);
        let min_cz = (bounds.min[2] - xzw_margin).div_euclid(POISSON_CELL_XZW);
        let max_cz = (bounds.max[2] + xzw_margin).div_euclid(POISSON_CELL_XZW);
        let min_cw = (bounds.min[3] - xzw_margin).div_euclid(POISSON_CELL_XZW);
        let max_cw = (bounds.max[3] + xzw_margin).div_euclid(POISSON_CELL_XZW);

        for cw in min_cw..=max_cw {
            for cz in min_cz..=max_cz {
                for cy in min_cy..=max_cy {
                    for cx in min_cx..=max_cx {
                        let Some(platform_bounds) =
                            self.sample_platform_bounds(cy, cx, cz, cw)
                        else {
                            continue;
                        };
                        if !intersects_xzw(bounds, platform_bounds) {
                            continue;
                        }
                        visitor(PlatformInstance {
                            level: cy,
                            cell_x: cx,
                            cell_z: cz,
                            cell_w: cw,
                            bounds: platform_bounds,
                        });
                    }
                }
            }
        }
    }

    fn insert_platform_uniform_nodes(&self, bounds: Aabb4i, tree: &mut RegionChunkTree) {
        let Some(material) = self.platform_voxel.as_ref() else {
            return;
        };
        let material = material.clone();
        self.for_each_platform_bounds_intersecting_query(bounds, |platform_bounds| {
            let core = RegionTreeCore {
                bounds: platform_bounds,
                kind: RegionNodeKind::Uniform(material.clone()),
                generator_version_hash: 0,
            };
            let _ = tree.splice_non_empty_core_in_bounds(platform_bounds, &core);
        });
    }

    fn insert_procgen_chunks_for_bounds(&self, bounds: Aabb4i, tree: &mut RegionChunkTree) {
        if !self.procgen_structures || !bounds.is_valid() {
            return;
        }
        let (local_min_y, local_max_y) = procgen::structure_chunk_y_bounds();
        // Procgen Y for platform at surface s: [s + 1 + local_min_y, s + 1 + local_max_y]
        // To intersect query [qmin_y, qmax_y]:
        //   s + 1 + local_min_y <= qmax_y AND s + 1 + local_max_y >= qmin_y
        //   s in [qmin_y - 1 - local_max_y, qmax_y - 1 - local_min_y]
        let surface_min = bounds.min[1] - 1 - local_max_y;
        let surface_max = bounds.max[1] - 1 - local_min_y;
        if surface_min > surface_max {
            return;
        }
        let min_cy = surface_min.div_euclid(POISSON_CELL_Y);
        let max_cy = surface_max.div_euclid(POISSON_CELL_Y);
        self.for_each_platform_instance_in_bounds(bounds, min_cy, max_cy, |platform| {
            self.insert_procgen_chunks_for_platform(bounds, platform, tree);
        });
    }

    fn insert_procgen_chunks_for_platform(
        &self,
        query_bounds: Aabb4i,
        platform: PlatformInstance,
        tree: &mut RegionChunkTree,
    ) {
        let procgen_frame_offset = procgen_frame_offset_xzw_chunks(self.world_seed, platform);
        let local_query_bounds =
            query_bounds_local_to_platform(query_bounds, platform, procgen_frame_offset);
        let Some(local_query_bounds) = local_query_bounds else {
            return;
        };
        let procgen_seed = hash_platform_cell(
            self.world_seed,
            platform.level,
            platform.cell_x,
            platform.cell_z,
            platform.cell_w,
            PLATFORM_HASH_SALT_PROCGEN_SEED,
        );

        // Generate per-placement structure chunk data (each placement isolated).
        let placement_data = procgen::generate_structure_placements_for_bounds(
            procgen_seed,
            local_query_bounds,
            self.procgen_keepout_cells(),
        );

        let platform_material = self.platform_voxel.as_ref().map(|v| self.content_registry.block_material_token(v.namespace, v.block_type));

        let y_offset = platform_y_offset(&platform);

        for placement in placement_data {
            if placement.chunks.is_empty() {
                continue;
            }

            let Some((effective_bounds, palette, indices, block_palette)) =
                build_chunk_array_from_placement(
                    &placement.chunks,
                    &procgen_frame_offset,
                    y_offset,
                    platform_material,
                )
            else {
                continue;
            };

            let chunk_array = match ChunkArrayData::from_dense_indices_with_block_palette(
                effective_bounds,
                palette,
                indices,
                Some(0), // default = Empty (palette index 0)
                block_palette,
            ) {
                Ok(ca) => ca,
                Err(_) => continue,
            };

            let core = RegionTreeCore {
                bounds: effective_bounds,
                kind: RegionNodeKind::ChunkArray(chunk_array),
                generator_version_hash: 0,
            };
            let _ = tree.splice_non_empty_core_in_bounds(effective_bounds, &core);
        }

        // Mazes still use the per-chunk path.
        self.insert_maze_chunks_for_platform(
            procgen_seed,
            local_query_bounds,
            &platform,
            procgen_frame_offset,
            tree,
        );
    }

    fn insert_maze_chunks_for_platform(
        &self,
        procgen_seed: u64,
        local_query_bounds: Aabb4i,
        platform: &PlatformInstance,
        procgen_frame_offset: [i32; 3],
        tree: &mut RegionChunkTree,
    ) {
        let placement_data = procgen::generate_maze_placements_for_bounds(
            procgen_seed,
            local_query_bounds,
        );

        let platform_material = self.platform_voxel.as_ref().map(|v| self.content_registry.block_material_token(v.namespace, v.block_type));
        let y_offset = platform_y_offset(platform);

        for placement in placement_data {
            if placement.chunks.is_empty() {
                continue;
            }

            let Some((effective_bounds, palette, indices, block_palette)) =
                build_chunk_array_from_placement(
                    &placement.chunks,
                    &procgen_frame_offset,
                    y_offset,
                    platform_material,
                )
            else {
                continue;
            };

            let chunk_array = match ChunkArrayData::from_dense_indices_with_block_palette(
                effective_bounds,
                palette,
                indices,
                Some(0), // default = Empty (palette index 0)
                block_palette,
            ) {
                Ok(ca) => ca,
                Err(_) => continue,
            };

            let core = RegionTreeCore {
                bounds: effective_bounds,
                kind: RegionNodeKind::ChunkArray(chunk_array),
                generator_version_hash: 0,
            };
            let _ = tree.splice_non_empty_core_in_bounds(effective_bounds, &core);
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
        self.insert_platform_uniform_nodes(bounds, &mut tree);
        self.insert_procgen_chunks_for_bounds(bounds, &mut tree);
        Arc::new(tree.root().cloned().unwrap_or(RegionTreeCore {
            bounds,
            kind: RegionNodeKind::Empty,
            generator_version_hash: 0,
        }))
    }
}

impl WorldField for MassivePlatformsWorldGenerator {
    fn query_region_core(&self, query: QueryVolume, detail: QueryDetail) -> Arc<RegionTreeCore> {
        MassivePlatformsWorldGenerator::query_region_core(self, query, detail)
    }
}

fn platform_voxel_from_base_kind(base_kind: BaseWorldKind) -> Option<BlockData> {
    match base_kind {
        BaseWorldKind::MassivePlatforms { material } if !material.is_air() => Some(material),
        BaseWorldKind::MassivePlatforms { .. }
        | BaseWorldKind::FlatFloor { .. }
        | BaseWorldKind::Empty => None,
    }
}

fn hash_platform_cell(
    world_seed: u64,
    level: i32,
    cell_x: i32,
    cell_z: i32,
    cell_w: i32,
    salt: u64,
) -> u64 {
    let mut state = world_seed ^ salt;
    state = mix_u64(state ^ ((level as i64 as u64).wrapping_mul(0x9e37_79b9_7f4a_7c15)));
    state = mix_u64(state ^ ((cell_x as i64 as u64).wrapping_mul(0xbf58_476d_1ce4_e5b9)));
    state = mix_u64(state ^ ((cell_z as i64 as u64).wrapping_mul(0x94d0_49bb_1331_11eb)));
    mix_u64(state ^ ((cell_w as i64 as u64).wrapping_mul(0x369d_ea0f_31a5_3f85)))
}

fn mix_u64(mut value: u64) -> u64 {
    value ^= value >> 30;
    value = value.wrapping_mul(0xbf58_476d_1ce4_e5b9);
    value ^= value >> 27;
    value = value.wrapping_mul(0x94d0_49bb_1331_11eb);
    value ^ (value >> 31)
}

fn is_origin_cell(cx: i32, cy: i32, cz: i32, cw: i32) -> bool {
    cx == 0 && cy == 0 && cz == 0 && cw == 0
}

/// Returns the candidate position [x, y, z, w] in chunk coordinates for a Poisson cell.
fn candidate_position_xyzw(seed: u64, cx: i32, cy: i32, cz: i32, cw: i32) -> [i32; 4] {
    let hx = hash_platform_cell(seed, cy, cx, cz, cw, SALT_CANDIDATE_X);
    let hy = hash_platform_cell(seed, cy, cx, cz, cw, SALT_CANDIDATE_Y);
    let hz = hash_platform_cell(seed, cy, cx, cz, cw, SALT_CANDIDATE_Z);
    let hw = hash_platform_cell(seed, cy, cx, cz, cw, SALT_CANDIDATE_W);
    [
        cx * POISSON_CELL_XZW + (hx % POISSON_CELL_XZW as u64) as i32,
        cy * POISSON_CELL_Y + (hy % POISSON_CELL_Y as u64) as i32,
        cz * POISSON_CELL_XZW + (hz % POISSON_CELL_XZW as u64) as i32,
        cw * POISSON_CELL_XZW + (hw % POISSON_CELL_XZW as u64) as i32,
    ]
}

/// Priority for tie-breaking when two candidates conflict.
fn cell_priority(seed: u64, cx: i32, cy: i32, cz: i32, cw: i32) -> u64 {
    hash_platform_cell(seed, cy, cx, cz, cw, SALT_PRIORITY)
}

/// Anisotropic squared distance with Y scaled up.
fn scaled_distance_sq(a: [i32; 4], b: [i32; 4]) -> i64 {
    let dx = (a[0] as i64) - (b[0] as i64);
    let dy = ((a[1] as i64) - (b[1] as i64)) * POISSON_Y_SCALE;
    let dz = (a[2] as i64) - (b[2] as i64);
    let dw = (a[3] as i64) - (b[3] as i64);
    dx * dx + dy * dy + dz * dz + dw * dw
}

/// Returns true if the candidate at cell (cx,cy,cz,cw) is accepted by the Poisson-disk filter.
/// Checks 80 neighboring cells (3^4 - 1). A candidate is rejected if a neighbor within the
/// minimum distance has higher priority (or equal priority with a smaller cell tuple).
fn is_candidate_accepted(seed: u64, cx: i32, cy: i32, cz: i32, cw: i32) -> bool {
    if is_origin_cell(cx, cy, cz, cw) {
        return true;
    }
    let my_pos = candidate_position_xyzw(seed, cx, cy, cz, cw);
    let my_pri = cell_priority(seed, cx, cy, cz, cw);
    for dw in -1i32..=1 {
        for dz in -1i32..=1 {
            for dy in -1i32..=1 {
                for dx in -1i32..=1 {
                    if dx == 0 && dy == 0 && dz == 0 && dw == 0 {
                        continue;
                    }
                    let nx = cx + dx;
                    let ny = cy + dy;
                    let nz = cz + dz;
                    let nw = cw + dw;
                    let neighbor_pos = if is_origin_cell(nx, ny, nz, nw) {
                        ORIGIN_ANCHOR_POS
                    } else {
                        candidate_position_xyzw(seed, nx, ny, nz, nw)
                    };
                    if scaled_distance_sq(my_pos, neighbor_pos) < POISSON_MIN_DIST_SQ {
                        // Origin anchor has infinite priority.
                        if is_origin_cell(nx, ny, nz, nw) {
                            return false;
                        }
                        let neighbor_pri = cell_priority(seed, nx, ny, nz, nw);
                        if neighbor_pri > my_pri
                            || (neighbor_pri == my_pri
                                && (ny, nx, nz, nw) < (cy, cx, cz, cw))
                        {
                            return false;
                        }
                    }
                }
            }
        }
    }
    true
}

fn platform_y_offset(platform: &PlatformInstance) -> i32 {
    platform.bounds.max[1] + 1
}

#[cfg(test)]
fn chunk_pos_from_local_to_platform(
    local: ChunkPos,
    platform: &PlatformInstance,
    procgen_frame_offset_xzw: [i32; 3],
) -> ChunkPos {
    let y_offset = platform_y_offset(platform);
    ChunkPos::new(
        local.x.saturating_sub(procgen_frame_offset_xzw[0]),
        local.y + y_offset,
        local.z.saturating_sub(procgen_frame_offset_xzw[1]),
        local.w.saturating_sub(procgen_frame_offset_xzw[2]),
    )
}

fn query_bounds_local_to_platform(
    query_bounds: Aabb4i,
    platform: PlatformInstance,
    procgen_frame_offset_xzw: [i32; 3],
) -> Option<Aabb4i> {
    let y_offset = platform_y_offset(&platform);
    let local = Aabb4i::new(
        [
            query_bounds.min[0]
                .max(platform.bounds.min[0])
                .saturating_add(procgen_frame_offset_xzw[0]),
            query_bounds.min[1] - y_offset,
            query_bounds.min[2]
                .max(platform.bounds.min[2])
                .saturating_add(procgen_frame_offset_xzw[1]),
            query_bounds.min[3]
                .max(platform.bounds.min[3])
                .saturating_add(procgen_frame_offset_xzw[2]),
        ],
        [
            query_bounds.max[0]
                .min(platform.bounds.max[0])
                .saturating_add(procgen_frame_offset_xzw[0]),
            query_bounds.max[1] - y_offset,
            query_bounds.max[2]
                .min(platform.bounds.max[2])
                .saturating_add(procgen_frame_offset_xzw[1]),
            query_bounds.max[3]
                .min(platform.bounds.max[3])
                .saturating_add(procgen_frame_offset_xzw[2]),
        ],
    );
    local.is_valid().then_some(local)
}

fn procgen_frame_offset_xzw_chunks(world_seed: u64, platform: PlatformInstance) -> [i32; 3] {
    let hash = hash_platform_cell(
        world_seed,
        platform.level,
        platform.cell_x,
        platform.cell_z,
        platform.cell_w,
        PLATFORM_HASH_SALT_PROCGEN_FRAME,
    );
    let x = PLATFORM_PROCGEN_FRAME_BASE_OFFSET_CHUNKS + ((hash & 0xfff) as i32);
    let z = PLATFORM_PROCGEN_FRAME_BASE_OFFSET_CHUNKS + (((hash >> 12) & 0xfff) as i32);
    let w = PLATFORM_PROCGEN_FRAME_BASE_OFFSET_CHUNKS + (((hash >> 24) & 0xfff) as i32);
    [x, z, w]
}

fn intersects_xzw(a: Aabb4i, b: Aabb4i) -> bool {
    a.min[0] <= b.max[0]
        && b.min[0] <= a.max[0]
        && a.min[2] <= b.max[2]
        && b.min[2] <= a.max[2]
        && a.min[3] <= b.max[3]
        && b.min[3] <= a.max[3]
}

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
    intersection.is_valid().then_some(intersection)
}

fn payload_from_chunk_compact(chunk: &DenseChunk) -> ChunkPayload {
    let materials: Vec<u16> = chunk.to_vec();
    ChunkPayload::from_dense_materials_compact(&materials)
        .unwrap_or(ChunkPayload::Dense16 { materials })
}

#[cfg(test)]
fn canonicalize_payload(resolved: ResolvedChunkPayload) -> ResolvedChunkPayload {
    let Ok(dense) = resolved.payload.dense_materials() else {
        return resolved;
    };
    if dense.is_empty() {
        return resolved;
    }
    let first = dense[0];
    if dense.iter().all(|material| *material == first) {
        let block = resolved
            .block_palette
            .get(first as usize)
            .cloned()
            .unwrap_or(BlockData::AIR);
        if block.is_air() {
            ResolvedChunkPayload::empty()
        } else {
            ResolvedChunkPayload::uniform(block)
        }
    } else {
        resolved
    }
}

/// Build a ChunkArray from a placement's chunk data.
///
/// Converts local chunk positions to world positions, filters out chunks that only contain
/// the platform material (preserving the Uniform node), computes tight bounds, builds a
/// deduplicated palette, and returns the data ready for `ChunkArrayData::from_dense_indices`.
///
/// Returns `None` if all chunks match the platform uniform or are empty.
fn build_chunk_array_from_placement(
    chunks: &HashMap<[i32; 4], DenseChunk>,
    procgen_frame_offset: &[i32; 3],
    y_offset: i32,
    platform_material: Option<u16>,
) -> Option<(Aabb4i, Vec<ChunkPayload>, Vec<u16>, Vec<BlockData>)> {
    // Convert local chunk positions to world chunk positions, filtering out chunks
    // that only contain platform material (or air + platform material).
    let mut world_chunks: Vec<([i32; 4], ChunkPayload)> = Vec::new();
    for (local_pos, chunk) in chunks {
        let world_pos = [
            local_pos[0].saturating_sub(procgen_frame_offset[0]),
            local_pos[1] + y_offset,
            local_pos[2].saturating_sub(procgen_frame_offset[1]),
            local_pos[3].saturating_sub(procgen_frame_offset[2]),
        ];

        // Check if this chunk only contains voxels matching the platform material.
        // If so, skip it — the platform Uniform node already covers it.
        if let Some(plat_mat) = platform_material {
            let dominated = chunk.iter().all(|&v| {
                v == 0 || v == plat_mat
            });
            if dominated {
                continue;
            }
        }

        let payload = payload_from_chunk_compact(chunk);
        world_chunks.push((world_pos, payload));
    }

    if world_chunks.is_empty() {
        return None;
    }

    // Compute tight bounding box of remaining world chunk positions.
    let mut tight_min = [i32::MAX; 4];
    let mut tight_max = [i32::MIN; 4];
    for (pos, _) in &world_chunks {
        for axis in 0..4 {
            tight_min[axis] = tight_min[axis].min(pos[axis]);
            tight_max[axis] = tight_max[axis].max(pos[axis]);
        }
    }
    let effective_bounds = Aabb4i::new(tight_min, tight_max);
    if !effective_bounds.is_valid() {
        return None;
    }

    // Build deduplicated palette and dense index array.
    let dims = [
        (effective_bounds.max[0] - effective_bounds.min[0] + 1) as usize,
        (effective_bounds.max[1] - effective_bounds.min[1] + 1) as usize,
        (effective_bounds.max[2] - effective_bounds.min[2] + 1) as usize,
        (effective_bounds.max[3] - effective_bounds.min[3] + 1) as usize,
    ];
    let cell_count = dims[0] * dims[1] * dims[2] * dims[3];

    // Use an empty/air chunk as the default palette entry (index 0).
    let mut palette: Vec<ChunkPayload> = vec![ChunkPayload::Empty];
    let mut palette_map: HashMap<ChunkPayload, u16> =
        HashMap::new();
    palette_map.insert(ChunkPayload::Empty, 0);

    let mut indices = vec![0u16; cell_count];

    for (pos, payload) in world_chunks {
        let rel = [
            (pos[0] - effective_bounds.min[0]) as usize,
            (pos[1] - effective_bounds.min[1]) as usize,
            (pos[2] - effective_bounds.min[2]) as usize,
            (pos[3] - effective_bounds.min[3]) as usize,
        ];
        let idx = rel[3] * dims[2] * dims[1] * dims[0]
            + rel[2] * dims[1] * dims[0]
            + rel[1] * dims[0]
            + rel[0];

        let palette_idx = *palette_map.entry(payload.clone()).or_insert_with(|| {
            let new_idx = palette.len() as u16;
            palette.push(payload);
            new_idx
        });
        indices[idx] = palette_idx;
    }

    // Build block_palette from the distinct material IDs used across all chunk payloads.
    // Each ChunkPayload was built with raw material IDs as u16 values by payload_from_chunk_compact.
    // We need to remap those to proper block_palette indices.
    let mut block_palette = vec![BlockData::AIR];
    let mut mat_to_block_idx: HashMap<u16, u16> = HashMap::new();
    mat_to_block_idx.insert(0, 0); // AIR -> index 0

    // Collect all distinct material IDs from all chunk payloads.
    for entry in &palette {
        match entry {
            ChunkPayload::Uniform(mat) => {
                mat_to_block_idx.entry(*mat).or_insert_with(|| {
                    let idx = block_palette.len() as u16;
                    block_palette.push(BlockData::simple(0, *mat as u32));
                    idx
                });
            }
            ChunkPayload::Dense16 { materials } => {
                for mat in materials {
                    mat_to_block_idx.entry(*mat).or_insert_with(|| {
                        let idx = block_palette.len() as u16;
                        block_palette.push(BlockData::simple(0, *mat as u32));
                        idx
                    });
                }
            }
            ChunkPayload::PalettePacked { palette: pal, .. } => {
                for mat in pal {
                    mat_to_block_idx.entry(*mat).or_insert_with(|| {
                        let idx = block_palette.len() as u16;
                        block_palette.push(BlockData::simple(0, *mat as u32));
                        idx
                    });
                }
            }
            ChunkPayload::Empty => {}
        }
    }

    // Remap all chunk payloads from raw material IDs to block_palette indices.
    let remapped_palette: Vec<ChunkPayload> = palette
        .into_iter()
        .map(|entry| match entry {
            ChunkPayload::Uniform(mat) => {
                ChunkPayload::Uniform(*mat_to_block_idx.get(&mat).unwrap_or(&mat))
            }
            ChunkPayload::Dense16 { materials } => ChunkPayload::Dense16 {
                materials: materials
                    .iter()
                    .map(|mat| *mat_to_block_idx.get(mat).unwrap_or(mat))
                    .collect(),
            },
            ChunkPayload::PalettePacked {
                palette: pal,
                bit_width,
                packed_indices,
            } => ChunkPayload::PalettePacked {
                palette: pal
                    .iter()
                    .map(|mat| *mat_to_block_idx.get(mat).unwrap_or(mat))
                    .collect(),
                bit_width,
                packed_indices,
            },
            other => other,
        })
        .collect();

    Some((effective_bounds, remapped_palette, indices, block_palette))
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::shared::region_tree::collect_non_empty_chunks_from_core_in_bounds;
    use std::collections::HashMap;

    fn test_registry() -> Arc<ContentRegistry> {
        let mut registry = ContentRegistry::new();
        crate::builtin_content::register_builtin_content(&mut registry);
        Arc::new(registry)
    }

    fn payload_at_chunk(core: &RegionTreeCore, chunk_key: [i32; 4]) -> Option<ResolvedChunkPayload> {
        let bounds = Aabb4i::new(chunk_key, chunk_key);
        collect_non_empty_chunks_from_core_in_bounds(core, bounds)
            .into_iter()
            .find_map(|(key, payload)| (key == chunk_key).then_some(payload))
    }

    fn collect_uniform_bounds(core: &RegionTreeCore, out: &mut Vec<Aabb4i>) {
        match &core.kind {
            RegionNodeKind::Uniform(_) => out.push(core.bounds),
            RegionNodeKind::Branch(children) => {
                for child in children {
                    collect_uniform_bounds(child, out);
                }
            }
            _ => {}
        }
    }

    /// Find a non-empty platform chunk key for a given seed, suitable for query-invariance tests.
    fn find_platform_chunk_key(seed: u64) -> [i32; 4] {
        let generator = MassivePlatformsWorldGenerator::from_chunk_payloads(
            BaseWorldKind::MassivePlatforms {
                material: BlockData::simple(0, 11),
            },
            Vec::<([i32; 4], ChunkPayload)>::new(),
            seed,
            true,
            HashSet::new(),
            test_registry(),
        );
        let bounds = Aabb4i::new([-96, -16, -96, -96], [96, 48, 96, 96]);
        let core = generator.query_region_core(QueryVolume { bounds }, QueryDetail::Exact);
        let chunks = collect_non_empty_chunks_from_core_in_bounds(core.as_ref(), bounds);
        // Find a chunk that's NOT in the origin anchor (to test non-trivial platforms).
        let origin_bounds = generator
            .sample_platform_bounds(0, 0, 0, 0)
            .unwrap();
        for (key, _) in &chunks {
            if key[0] < origin_bounds.min[0]
                || key[0] > origin_bounds.max[0]
                || key[2] < origin_bounds.min[2]
                || key[2] > origin_bounds.max[2]
                || key[3] < origin_bounds.min[3]
                || key[3] > origin_bounds.max[3]
            {
                return *key;
            }
        }
        // Fallback to any non-empty chunk.
        chunks[0].0
    }

    #[test]
    fn invalid_bounds_query_returns_empty() {
        let generator = MassivePlatformsWorldGenerator::from_chunk_payloads(
            BaseWorldKind::MassivePlatforms {
                material: BlockData::simple(0, 11),
            },
            Vec::<([i32; 4], ChunkPayload)>::new(),
            7,
            false,
            HashSet::new(),
            test_registry(),
        );
        let bounds = Aabb4i::new([3, 0, 0, 0], [2, 0, 0, 0]);
        let core = generator.query_region_core(QueryVolume { bounds }, QueryDetail::Exact);
        assert_eq!(core.bounds, bounds);
        assert!(matches!(core.kind, RegionNodeKind::Empty));
    }

    #[test]
    fn massive_platform_world_contains_large_uniform_nodes() {
        let generator = MassivePlatformsWorldGenerator::from_chunk_payloads(
            BaseWorldKind::MassivePlatforms {
                material: BlockData::simple(0, 11),
            },
            Vec::<([i32; 4], ChunkPayload)>::new(),
            1337,
            false,
            HashSet::new(),
            test_registry(),
        );
        let bounds = Aabb4i::new([-96, -2, -96, -96], [96, 30, 96, 96]);
        let core = generator.query_region_core(QueryVolume { bounds }, QueryDetail::Exact);
        let mut uniform_bounds = Vec::new();
        collect_uniform_bounds(core.as_ref(), &mut uniform_bounds);
        assert!(!uniform_bounds.is_empty());
        assert!(uniform_bounds.iter().any(|platform| {
            let dx = platform.max[0] - platform.min[0] + 1;
            let dz = platform.max[2] - platform.min[2] + 1;
            let dw = platform.max[3] - platform.min[3] + 1;
            dx >= 20 && dz >= 20 && dw >= 20
        }));
    }

    #[test]
    fn origin_anchor_platform_exists_with_air_above_spawn() {
        let generator = MassivePlatformsWorldGenerator::from_chunk_payloads(
            BaseWorldKind::MassivePlatforms {
                material: BlockData::simple(0, 11),
            },
            Vec::<([i32; 4], ChunkPayload)>::new(),
            1337,
            false,
            HashSet::new(),
            test_registry(),
        );
        let bounds = Aabb4i::new([-1, -1, -1, -1], [1, 1, 1, 1]);
        let core = generator.query_region_core(QueryVolume { bounds }, QueryDetail::Exact);
        let non_empty = collect_non_empty_chunks_from_core_in_bounds(core.as_ref(), bounds);
        assert!(non_empty.iter().any(|(key, _)| *key == [0, 0, 0, 0]));
        assert!(!non_empty.iter().any(|(key, _)| *key == [0, 1, 0, 0]));
    }

    #[test]
    fn origin_anchor_platform_can_host_procgen_structure_chunks() {
        let generator = MassivePlatformsWorldGenerator::from_chunk_payloads(
            BaseWorldKind::MassivePlatforms {
                material: BlockData::simple(0, 11),
            },
            Vec::<([i32; 4], ChunkPayload)>::new(),
            1337,
            true,
            HashSet::new(),
            test_registry(),
        );
        let (local_min_y, local_max_y) = procgen::structure_chunk_y_bounds();
        let platform = PlatformInstance {
            level: 0,
            cell_x: 0,
            cell_z: 0,
            cell_w: 0,
            bounds: generator
                .sample_platform_bounds(0, 0, 0, 0)
                .expect("origin anchor platform must exist"),
        };
        let probe = Aabb4i::new(
            [
                platform.bounds.min[0],
                local_min_y,
                platform.bounds.min[2],
                platform.bounds.min[3],
            ],
            [
                platform.bounds.max[0],
                local_max_y,
                platform.bounds.max[2],
                platform.bounds.max[3],
            ],
        );
        let local_probe = query_bounds_local_to_platform(
            probe,
            platform,
            procgen_frame_offset_xzw_chunks(generator.world_seed, platform),
        )
        .expect("origin platform probe should be valid");
        let procgen_seed = hash_platform_cell(
            generator.world_seed,
            platform.level,
            platform.cell_x,
            platform.cell_z,
            platform.cell_w,
            PLATFORM_HASH_SALT_PROCGEN_SEED,
        );
        let candidates = procgen::structure_chunk_positions_for_bounds_with_keepout(
            procgen_seed,
            local_probe,
            None,
        );
        assert!(
            !candidates.is_empty(),
            "expected at least one procgen chunk candidate on origin platform"
        );
    }

    #[test]
    fn procgen_structure_spawning_reaches_multiple_y_levels() {
        let registry = test_registry();
        let seeded = MassivePlatformsWorldGenerator::from_chunk_payloads(
            BaseWorldKind::MassivePlatforms {
                material: BlockData::simple(0, 11),
            },
            Vec::<([i32; 4], ChunkPayload)>::new(),
            1337,
            true,
            HashSet::new(),
            registry.clone(),
        );
        let no_procgen = MassivePlatformsWorldGenerator::from_chunk_payloads(
            BaseWorldKind::MassivePlatforms {
                material: BlockData::simple(0, 11),
            },
            Vec::<([i32; 4], ChunkPayload)>::new(),
            1337,
            false,
            HashSet::new(),
            registry,
        );
        let (local_min_y, local_max_y) = procgen::structure_chunk_y_bounds();
        let search = Aabb4i::new([-96, -64, -96, -96], [96, 96, 96, 96]);
        let mut validated = false;
        // Search cells at cy >= 1 to find platforms above level 0.
        let min_cy = 1;
        let max_cy = 8;
        seeded.for_each_platform_instance_in_bounds(search, min_cy, max_cy, |platform| {
            if validated {
                return;
            }
            let y_offset = platform_y_offset(&platform);
            let frame_offset = procgen_frame_offset_xzw_chunks(seeded.world_seed, platform);
            // Build probe in world coords, then convert to frame-shifted local space.
            let world_probe = Aabb4i::new(
                [
                    platform.bounds.min[0],
                    y_offset + local_min_y,
                    platform.bounds.min[2],
                    platform.bounds.min[3],
                ],
                [
                    platform.bounds.max[0],
                    y_offset + local_max_y,
                    platform.bounds.max[2],
                    platform.bounds.max[3],
                ],
            );
            let Some(local_probe) =
                query_bounds_local_to_platform(world_probe, platform, frame_offset)
            else {
                return;
            };
            let seed = hash_platform_cell(
                seeded.world_seed,
                platform.level,
                platform.cell_x,
                platform.cell_z,
                platform.cell_w,
                PLATFORM_HASH_SALT_PROCGEN_SEED,
            );
            let Some(local_chunk) =
                procgen::structure_chunk_positions_for_bounds_with_keepout(seed, local_probe, None)
                    .into_iter()
                    .next()
            else {
                return;
            };
            let global_chunk = chunk_pos_from_local_to_platform(
                local_chunk,
                &platform,
                frame_offset,
            );
            // Verify the global chunk is above the origin platform surface.
            if global_chunk.y < POISSON_CELL_Y {
                return;
            }

            let chunk_bounds = Aabb4i::new(
                [
                    global_chunk.x,
                    global_chunk.y,
                    global_chunk.z,
                    global_chunk.w,
                ],
                [
                    global_chunk.x,
                    global_chunk.y,
                    global_chunk.z,
                    global_chunk.w,
                ],
            );
            let with_core = seeded.query_region_core(
                QueryVolume {
                    bounds: chunk_bounds,
                },
                QueryDetail::Exact,
            );
            let without_core = no_procgen.query_region_core(
                QueryVolume {
                    bounds: chunk_bounds,
                },
                QueryDetail::Exact,
            );
            let with_chunks =
                collect_non_empty_chunks_from_core_in_bounds(with_core.as_ref(), chunk_bounds);
            let without_chunks =
                collect_non_empty_chunks_from_core_in_bounds(without_core.as_ref(), chunk_bounds);
            let target_key = [
                global_chunk.x,
                global_chunk.y,
                global_chunk.z,
                global_chunk.w,
            ];
            let with_payload = with_chunks.iter().find_map(|(key, payload)| {
                (*key == target_key).then_some(canonicalize_payload(payload.clone()))
            });
            let without_payload = without_chunks.iter().find_map(|(key, payload)| {
                (*key == target_key).then_some(canonicalize_payload(payload.clone()))
            });
            if with_payload != without_payload {
                validated = true;
            }
        });
        assert!(validated);
    }

    #[test]
    fn procgen_chunks_are_anchored_to_some_platform_volume() {
        let registry = test_registry();
        let generator = MassivePlatformsWorldGenerator::from_chunk_payloads(
            BaseWorldKind::MassivePlatforms {
                material: BlockData::simple(0, 11),
            },
            Vec::<([i32; 4], ChunkPayload)>::new(),
            1337,
            true,
            HashSet::new(),
            registry.clone(),
        );
        let no_procgen = MassivePlatformsWorldGenerator::from_chunk_payloads(
            BaseWorldKind::MassivePlatforms {
                material: BlockData::simple(0, 11),
            },
            Vec::<([i32; 4], ChunkPayload)>::new(),
            1337,
            false,
            HashSet::new(),
            registry,
        );
        let bounds = Aabb4i::new([-24, -8, -24, -24], [24, 36, 24, 24]);
        let with_structures =
            generator.query_region_core(QueryVolume { bounds }, QueryDetail::Exact);
        let without_structures =
            no_procgen.query_region_core(QueryVolume { bounds }, QueryDetail::Exact);

        let mut with_map = HashMap::new();
        for (key, payload) in
            collect_non_empty_chunks_from_core_in_bounds(with_structures.as_ref(), bounds)
        {
            with_map.insert(key, canonicalize_payload(payload));
        }
        let mut without_map = HashMap::new();
        for (key, payload) in
            collect_non_empty_chunks_from_core_in_bounds(without_structures.as_ref(), bounds)
        {
            without_map.insert(key, canonicalize_payload(payload));
        }

        let (local_min_y, local_max_y) = procgen::structure_chunk_y_bounds();
        let mut changed_keys = Vec::new();
        for (key, payload) in &with_map {
            let unchanged = without_map
                .get(key)
                .map(|base| base == payload)
                .unwrap_or(false);
            if !unchanged {
                changed_keys.push(*key);
            }
        }
        assert!(!changed_keys.is_empty());

        for key in changed_keys.into_iter().take(64) {
            let key_bounds = Aabb4i::new(key, key);
            // Find platforms whose procgen Y extent covers this chunk.
            // Procgen Y = [y_offset + local_min_y, y_offset + local_max_y]
            // y_offset = surface + 1, surface = candidate Y.
            // We need: y_offset + local_min_y <= key[1] <= y_offset + local_max_y
            // So: surface in [key[1] - 1 - local_max_y, key[1] - 1 - local_min_y]
            let surface_min = key[1] - 1 - local_max_y;
            let surface_max = key[1] - 1 - local_min_y;
            let min_cy = surface_min.div_euclid(POISSON_CELL_Y);
            let max_cy = surface_max.div_euclid(POISSON_CELL_Y);
            let mut anchored = false;
            generator.for_each_platform_instance_in_bounds(
                key_bounds,
                min_cy,
                max_cy,
                |platform| {
                    let y_offset = platform_y_offset(&platform);
                    let in_horizontal = key[0] >= platform.bounds.min[0]
                        && key[0] <= platform.bounds.max[0]
                        && key[2] >= platform.bounds.min[2]
                        && key[2] <= platform.bounds.max[2]
                        && key[3] >= platform.bounds.min[3]
                        && key[3] <= platform.bounds.max[3];
                    let in_vertical =
                        key[1] >= y_offset + local_min_y && key[1] <= y_offset + local_max_y;
                    if in_horizontal && in_vertical {
                        anchored = true;
                    }
                },
            );
            assert!(anchored, "unanchored structure chunk at {:?}", key);
        }
    }

    #[test]
    fn chunk_sampling_is_query_volume_invariant_across_platform_y_boundaries() {
        let seed = 0xD1A6_2026;
        let generator = MassivePlatformsWorldGenerator::from_chunk_payloads(
            BaseWorldKind::MassivePlatforms {
                material: BlockData::simple(0, 11),
            },
            Vec::<([i32; 4], ChunkPayload)>::new(),
            seed,
            true,
            HashSet::new(),
            test_registry(),
        );

        let chunk_key = find_platform_chunk_key(seed);
        let single_bounds = Aabb4i::new(chunk_key, chunk_key);
        let neighborhood_bounds = Aabb4i::new(
            [
                chunk_key[0] - 1,
                chunk_key[1] - 1,
                chunk_key[2] - 1,
                chunk_key[3] - 1,
            ],
            [
                chunk_key[0] + 1,
                chunk_key[1] + 1,
                chunk_key[2] + 1,
                chunk_key[3] + 1,
            ],
        );
        let wide_bounds = Aabb4i::new(
            [
                chunk_key[0] - 8,
                chunk_key[1] - 8,
                chunk_key[2] - 8,
                chunk_key[3] - 8,
            ],
            [
                chunk_key[0] + 8,
                chunk_key[1] + 8,
                chunk_key[2] + 8,
                chunk_key[3] + 8,
            ],
        );

        let single = generator.query_region_core(
            QueryVolume {
                bounds: single_bounds,
            },
            QueryDetail::Exact,
        );
        let neighborhood = generator.query_region_core(
            QueryVolume {
                bounds: neighborhood_bounds,
            },
            QueryDetail::Exact,
        );
        let wide = generator.query_region_core(
            QueryVolume {
                bounds: wide_bounds,
            },
            QueryDetail::Exact,
        );

        let expected = payload_at_chunk(single.as_ref(), chunk_key);
        assert_eq!(
            payload_at_chunk(neighborhood.as_ref(), chunk_key),
            expected,
            "chunk payload changed between single-cell and neighborhood query",
        );
        assert_eq!(
            payload_at_chunk(wide.as_ref(), chunk_key),
            expected,
            "chunk payload changed between single-cell and wide query",
        );
    }
}
