use super::{QueryDetail, QueryVolume, WorldField};
use crate::server::procgen;
use crate::shared::chunk_payload::ChunkPayload;
use crate::shared::region_tree::{
    chunk_key_from_chunk_pos, RegionChunkTree, RegionNodeKind, RegionTreeCore,
};
use crate::shared::spatial::Aabb4i;
use crate::shared::voxel::{BaseWorldKind, ChunkPos, VoxelType, CHUNK_VOLUME};
use std::collections::HashSet;
use std::sync::Arc;

type DenseChunk = [VoxelType; CHUNK_VOLUME];

const PLATFORM_GRID_STRIDE_XZW_CHUNKS: i32 = 48;
const PLATFORM_LEVEL_STRIDE_Y_CHUNKS: i32 = 12;
const PLATFORM_THICKNESS_CHUNKS: i32 = 2;
const PLATFORM_HALF_EXTENT_MIN_CHUNKS: i32 = 12;
const PLATFORM_HALF_EXTENT_MAX_CHUNKS: i32 = 20;
const PLATFORM_CENTER_JITTER_CHUNKS: i32 = 8;
const PLATFORM_PRESENCE_PERCENT: u64 = 86;
const ORIGIN_ANCHOR_LEVEL: i32 = 0;
const ORIGIN_ANCHOR_CELL_X: i32 = 0;
const ORIGIN_ANCHOR_CELL_Z: i32 = 0;
const ORIGIN_ANCHOR_CELL_W: i32 = 0;
const ORIGIN_ANCHOR_HALF_EXTENT_CHUNKS: i32 = 18;

const PLATFORM_HASH_SALT_PRESENCE: u64 = 0x91f4_2b7e_87f1_3a55;
const PLATFORM_HASH_SALT_EXTENT_X: u64 = 0x58d7_6c29_2a0b_4cd1;
const PLATFORM_HASH_SALT_EXTENT_Z: u64 = 0x7ef8_18a3_02e5_2b3f;
const PLATFORM_HASH_SALT_EXTENT_W: u64 = 0x4cc2_743e_91ad_0627;
const PLATFORM_HASH_SALT_JITTER_X: u64 = 0x0a1e_d952_75c4_6619;
const PLATFORM_HASH_SALT_JITTER_Z: u64 = 0x8f63_d4b2_3c14_20e3;
const PLATFORM_HASH_SALT_JITTER_W: u64 = 0x3ce7_b890_5d61_489f;
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

#[derive(Debug)]
pub struct MassivePlatformsWorldGenerator {
    platform_voxel: Option<VoxelType>,
    world_seed: u64,
    procgen_structures: bool,
    blocked_cells: HashSet<procgen::StructureCell>,
}

impl MassivePlatformsWorldGenerator {
    pub fn from_chunk_payloads(
        base_kind: BaseWorldKind,
        _chunk_payloads: impl IntoIterator<Item = ([i32; 4], ChunkPayload)>,
        world_seed: u64,
        procgen_structures: bool,
        blocked_cells: HashSet<procgen::StructureCell>,
    ) -> Self {
        Self {
            platform_voxel: platform_voxel_from_base_kind(base_kind),
            world_seed,
            procgen_structures,
            blocked_cells,
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
        level: i32,
        cell_x: i32,
        cell_z: i32,
        cell_w: i32,
    ) -> Option<Aabb4i> {
        let material = self.platform_voxel?;
        if material.is_air() {
            return None;
        }

        if level == ORIGIN_ANCHOR_LEVEL
            && cell_x == ORIGIN_ANCHOR_CELL_X
            && cell_z == ORIGIN_ANCHOR_CELL_Z
            && cell_w == ORIGIN_ANCHOR_CELL_W
        {
            let min_y = platform_min_chunk_y_for_level(level);
            let max_y = min_y + PLATFORM_THICKNESS_CHUNKS - 1;
            return Some(Aabb4i::new(
                [
                    -ORIGIN_ANCHOR_HALF_EXTENT_CHUNKS,
                    min_y,
                    -ORIGIN_ANCHOR_HALF_EXTENT_CHUNKS,
                    -ORIGIN_ANCHOR_HALF_EXTENT_CHUNKS,
                ],
                [
                    ORIGIN_ANCHOR_HALF_EXTENT_CHUNKS - 1,
                    max_y,
                    ORIGIN_ANCHOR_HALF_EXTENT_CHUNKS - 1,
                    ORIGIN_ANCHOR_HALF_EXTENT_CHUNKS - 1,
                ],
            ));
        }

        let presence_hash = hash_platform_cell(
            self.world_seed,
            level,
            cell_x,
            cell_z,
            cell_w,
            PLATFORM_HASH_SALT_PRESENCE,
        );
        if presence_hash % 100 >= PLATFORM_PRESENCE_PERCENT {
            return None;
        }

        let extent_span =
            (PLATFORM_HALF_EXTENT_MAX_CHUNKS - PLATFORM_HALF_EXTENT_MIN_CHUNKS + 1).max(1) as u64;
        let half_x = PLATFORM_HALF_EXTENT_MIN_CHUNKS
            + (hash_platform_cell(
                self.world_seed,
                level,
                cell_x,
                cell_z,
                cell_w,
                PLATFORM_HASH_SALT_EXTENT_X,
            ) % extent_span) as i32;
        let half_z = PLATFORM_HALF_EXTENT_MIN_CHUNKS
            + (hash_platform_cell(
                self.world_seed,
                level,
                cell_x,
                cell_z,
                cell_w,
                PLATFORM_HASH_SALT_EXTENT_Z,
            ) % extent_span) as i32;
        let half_w = PLATFORM_HALF_EXTENT_MIN_CHUNKS
            + (hash_platform_cell(
                self.world_seed,
                level,
                cell_x,
                cell_z,
                cell_w,
                PLATFORM_HASH_SALT_EXTENT_W,
            ) % extent_span) as i32;

        let jitter_x = signed_jitter(
            hash_platform_cell(
                self.world_seed,
                level,
                cell_x,
                cell_z,
                cell_w,
                PLATFORM_HASH_SALT_JITTER_X,
            ),
            PLATFORM_CENTER_JITTER_CHUNKS,
        );
        let jitter_z = signed_jitter(
            hash_platform_cell(
                self.world_seed,
                level,
                cell_x,
                cell_z,
                cell_w,
                PLATFORM_HASH_SALT_JITTER_Z,
            ),
            PLATFORM_CENTER_JITTER_CHUNKS,
        );
        let jitter_w = signed_jitter(
            hash_platform_cell(
                self.world_seed,
                level,
                cell_x,
                cell_z,
                cell_w,
                PLATFORM_HASH_SALT_JITTER_W,
            ),
            PLATFORM_CENTER_JITTER_CHUNKS,
        );

        let center_x = cell_x * PLATFORM_GRID_STRIDE_XZW_CHUNKS
            + (PLATFORM_GRID_STRIDE_XZW_CHUNKS / 2)
            + jitter_x;
        let center_z = cell_z * PLATFORM_GRID_STRIDE_XZW_CHUNKS
            + (PLATFORM_GRID_STRIDE_XZW_CHUNKS / 2)
            + jitter_z;
        let center_w = cell_w * PLATFORM_GRID_STRIDE_XZW_CHUNKS
            + (PLATFORM_GRID_STRIDE_XZW_CHUNKS / 2)
            + jitter_w;
        let min_y = platform_min_chunk_y_for_level(level);
        let max_y = min_y + PLATFORM_THICKNESS_CHUNKS - 1;

        let platform_bounds = Aabb4i::new(
            [
                center_x - half_x,
                min_y,
                center_z - half_z,
                center_w - half_w,
            ],
            [
                center_x + half_x - 1,
                max_y,
                center_z + half_z - 1,
                center_w + half_w - 1,
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
        let min_level = (bounds.min[1] - (PLATFORM_THICKNESS_CHUNKS - 1))
            .div_euclid(PLATFORM_LEVEL_STRIDE_Y_CHUNKS);
        let max_level = bounds.max[1].div_euclid(PLATFORM_LEVEL_STRIDE_Y_CHUNKS);
        self.for_each_platform_instance_in_level_range(bounds, min_level, max_level, |platform| {
            let Some(intersection) = intersect_bounds(bounds, platform.bounds) else {
                return;
            };
            visitor(intersection);
        });
    }

    fn for_each_platform_instance_in_level_range<F>(
        &self,
        bounds: Aabb4i,
        min_level: i32,
        max_level: i32,
        mut visitor: F,
    ) where
        F: FnMut(PlatformInstance),
    {
        if !bounds.is_valid() || self.platform_voxel.is_none() || min_level > max_level {
            return;
        }

        let horizontal_margin = PLATFORM_HALF_EXTENT_MAX_CHUNKS + PLATFORM_CENTER_JITTER_CHUNKS + 1;
        let min_cell_x =
            (bounds.min[0] - horizontal_margin).div_euclid(PLATFORM_GRID_STRIDE_XZW_CHUNKS);
        let max_cell_x =
            (bounds.max[0] + horizontal_margin).div_euclid(PLATFORM_GRID_STRIDE_XZW_CHUNKS);
        let min_cell_z =
            (bounds.min[2] - horizontal_margin).div_euclid(PLATFORM_GRID_STRIDE_XZW_CHUNKS);
        let max_cell_z =
            (bounds.max[2] + horizontal_margin).div_euclid(PLATFORM_GRID_STRIDE_XZW_CHUNKS);
        let min_cell_w =
            (bounds.min[3] - horizontal_margin).div_euclid(PLATFORM_GRID_STRIDE_XZW_CHUNKS);
        let max_cell_w =
            (bounds.max[3] + horizontal_margin).div_euclid(PLATFORM_GRID_STRIDE_XZW_CHUNKS);

        for level in min_level..=max_level {
            for cell_w in min_cell_w..=max_cell_w {
                for cell_z in min_cell_z..=max_cell_z {
                    for cell_x in min_cell_x..=max_cell_x {
                        let Some(platform_bounds) =
                            self.sample_platform_bounds(level, cell_x, cell_z, cell_w)
                        else {
                            continue;
                        };
                        if !intersects_xzw(bounds, platform_bounds) {
                            continue;
                        }
                        visitor(PlatformInstance {
                            level,
                            cell_x,
                            cell_z,
                            cell_w,
                            bounds: platform_bounds,
                        });
                    }
                }
            }
        }
    }

    fn insert_platform_uniform_nodes(&self, bounds: Aabb4i, tree: &mut RegionChunkTree) {
        let Some(material) = self.platform_voxel else {
            return;
        };
        self.for_each_platform_bounds_intersecting_query(bounds, |platform_bounds| {
            let core = RegionTreeCore {
                bounds: platform_bounds,
                kind: RegionNodeKind::Uniform(u16::from(material.0)),
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
        let min_level = (bounds.min[1] - local_max_y).div_euclid(PLATFORM_LEVEL_STRIDE_Y_CHUNKS);
        let max_level = (bounds.max[1] - local_min_y).div_euclid(PLATFORM_LEVEL_STRIDE_Y_CHUNKS);
        self.for_each_platform_instance_in_level_range(bounds, min_level, max_level, |platform| {
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
        let candidates = procgen::structure_chunk_positions_for_bounds_with_keepout(
            procgen_seed,
            local_query_bounds,
            self.procgen_keepout_cells(),
        );
        for local_chunk_pos in candidates {
            let Some(structure_chunk) = procgen::generate_structure_chunk_with_keepout(
                procgen_seed,
                local_chunk_pos,
                self.procgen_keepout_cells(),
            ) else {
                continue;
            };
            let chunk_pos = chunk_pos_from_local_to_platform(
                local_chunk_pos,
                platform.level,
                procgen_frame_offset,
            );
            let key = chunk_key_from_chunk_pos(chunk_pos);
            let base_payload = tree.chunk_payload(key);
            let mut merged = base_payload
                .as_ref()
                .and_then(chunk_from_payload)
                .unwrap_or_else(empty_dense_chunk);
            merge_non_air_voxels(&mut merged, &structure_chunk);
            let payload = payload_from_chunk_compact(&merged);
            let is_same_as_base = base_payload
                .as_ref()
                .map(|base| {
                    canonicalize_payload(base.clone()) == canonicalize_payload(payload.clone())
                })
                .unwrap_or(false);
            if is_same_as_base {
                continue;
            }
            let _ = tree.set_chunk(key, Some(payload));
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
        Arc::new(tree.slice_non_empty_core_in_bounds(bounds))
    }
}

impl WorldField for MassivePlatformsWorldGenerator {
    fn query_region_core(&self, query: QueryVolume, detail: QueryDetail) -> Arc<RegionTreeCore> {
        MassivePlatformsWorldGenerator::query_region_core(self, query, detail)
    }
}

fn platform_voxel_from_base_kind(base_kind: BaseWorldKind) -> Option<VoxelType> {
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

fn signed_jitter(hash: u64, max_abs: i32) -> i32 {
    if max_abs <= 0 {
        return 0;
    }
    let span = ((max_abs * 2) + 1) as u64;
    (hash % span) as i32 - max_abs
}

fn platform_min_chunk_y_for_level(level: i32) -> i32 {
    level * PLATFORM_LEVEL_STRIDE_Y_CHUNKS - PLATFORM_THICKNESS_CHUNKS
}

fn platform_level_chunk_y_offset(level: i32) -> i32 {
    level * PLATFORM_LEVEL_STRIDE_Y_CHUNKS
}

fn chunk_pos_from_local_to_platform(
    local: ChunkPos,
    level: i32,
    procgen_frame_offset_xzw: [i32; 3],
) -> ChunkPos {
    ChunkPos::new(
        local.x.saturating_sub(procgen_frame_offset_xzw[0]),
        local.y + platform_level_chunk_y_offset(level),
        local.z.saturating_sub(procgen_frame_offset_xzw[1]),
        local.w.saturating_sub(procgen_frame_offset_xzw[2]),
    )
}

fn query_bounds_local_to_platform(
    query_bounds: Aabb4i,
    platform: PlatformInstance,
    procgen_frame_offset_xzw: [i32; 3],
) -> Option<Aabb4i> {
    let y_offset = platform_level_chunk_y_offset(platform.level);
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
    let materials: Vec<u16> = chunk.iter().map(|voxel| u16::from(voxel.0)).collect();
    ChunkPayload::from_dense_materials_compact(&materials)
        .unwrap_or(ChunkPayload::Dense16 { materials })
}

fn canonicalize_payload(payload: ChunkPayload) -> ChunkPayload {
    let payload = match payload {
        ChunkPayload::Empty => ChunkPayload::Uniform(0),
        other => other,
    };
    let Ok(dense) = payload.dense_materials() else {
        return payload;
    };
    if dense.is_empty() {
        return payload;
    }
    let first = dense[0];
    if dense.iter().all(|material| *material == first) {
        ChunkPayload::Uniform(first)
    } else {
        payload
    }
}

fn chunk_from_payload(payload: &ChunkPayload) -> Option<DenseChunk> {
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

fn empty_dense_chunk() -> DenseChunk {
    [VoxelType::AIR; CHUNK_VOLUME]
}

fn merge_non_air_voxels(dst: &mut DenseChunk, src: &DenseChunk) {
    for (idx, voxel) in src.iter().enumerate() {
        if voxel.is_air() {
            continue;
        }
        dst[idx] = *voxel;
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::shared::region_tree::collect_non_empty_chunks_from_core_in_bounds;
    use std::collections::HashMap;

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

    #[test]
    fn invalid_bounds_query_returns_empty() {
        let generator = MassivePlatformsWorldGenerator::from_chunk_payloads(
            BaseWorldKind::MassivePlatforms {
                material: VoxelType(11),
            },
            Vec::<([i32; 4], ChunkPayload)>::new(),
            7,
            false,
            HashSet::new(),
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
                material: VoxelType(11),
            },
            Vec::<([i32; 4], ChunkPayload)>::new(),
            1337,
            false,
            HashSet::new(),
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
                material: VoxelType(11),
            },
            Vec::<([i32; 4], ChunkPayload)>::new(),
            1337,
            false,
            HashSet::new(),
        );
        let bounds = Aabb4i::new([-1, -1, -1, -1], [1, 0, 1, 1]);
        let core = generator.query_region_core(QueryVolume { bounds }, QueryDetail::Exact);
        let non_empty = collect_non_empty_chunks_from_core_in_bounds(core.as_ref(), bounds);
        assert!(non_empty.iter().any(|(key, _)| *key == [0, -1, 0, 0]));
        assert!(!non_empty.iter().any(|(key, _)| *key == [0, 0, 0, 0]));
    }

    #[test]
    fn origin_anchor_platform_can_host_procgen_structure_chunks() {
        let generator = MassivePlatformsWorldGenerator::from_chunk_payloads(
            BaseWorldKind::MassivePlatforms {
                material: VoxelType(11),
            },
            Vec::<([i32; 4], ChunkPayload)>::new(),
            1337,
            true,
            HashSet::new(),
        );
        let (local_min_y, local_max_y) = procgen::structure_chunk_y_bounds();
        let platform = PlatformInstance {
            level: ORIGIN_ANCHOR_LEVEL,
            cell_x: ORIGIN_ANCHOR_CELL_X,
            cell_z: ORIGIN_ANCHOR_CELL_Z,
            cell_w: ORIGIN_ANCHOR_CELL_W,
            bounds: generator
                .sample_platform_bounds(
                    ORIGIN_ANCHOR_LEVEL,
                    ORIGIN_ANCHOR_CELL_X,
                    ORIGIN_ANCHOR_CELL_Z,
                    ORIGIN_ANCHOR_CELL_W,
                )
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
        let seeded = MassivePlatformsWorldGenerator::from_chunk_payloads(
            BaseWorldKind::MassivePlatforms {
                material: VoxelType(11),
            },
            Vec::<([i32; 4], ChunkPayload)>::new(),
            1337,
            true,
            HashSet::new(),
        );
        let no_procgen = MassivePlatformsWorldGenerator::from_chunk_payloads(
            BaseWorldKind::MassivePlatforms {
                material: VoxelType(11),
            },
            Vec::<([i32; 4], ChunkPayload)>::new(),
            1337,
            false,
            HashSet::new(),
        );
        let (local_min_y, local_max_y) = procgen::structure_chunk_y_bounds();
        let search = Aabb4i::new([-96, -64, -96, -96], [96, 96, 96, 96]);
        let mut validated = false;
        seeded.for_each_platform_instance_in_level_range(search, 1, 8, |platform| {
            if validated {
                return;
            }
            let local_probe = Aabb4i::new(
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
                platform.level,
                procgen_frame_offset_xzw_chunks(seeded.world_seed, platform),
            );
            if global_chunk.y < PLATFORM_LEVEL_STRIDE_Y_CHUNKS {
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
        let generator = MassivePlatformsWorldGenerator::from_chunk_payloads(
            BaseWorldKind::MassivePlatforms {
                material: VoxelType(11),
            },
            Vec::<([i32; 4], ChunkPayload)>::new(),
            1337,
            true,
            HashSet::new(),
        );
        let no_procgen = MassivePlatformsWorldGenerator::from_chunk_payloads(
            BaseWorldKind::MassivePlatforms {
                material: VoxelType(11),
            },
            Vec::<([i32; 4], ChunkPayload)>::new(),
            1337,
            false,
            HashSet::new(),
        );
        let bounds = Aabb4i::new([-24, -8, -24, 0], [24, 36, 24, 0]);
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
            let min_level = (key[1] - local_max_y).div_euclid(PLATFORM_LEVEL_STRIDE_Y_CHUNKS);
            let max_level = (key[1] - local_min_y).div_euclid(PLATFORM_LEVEL_STRIDE_Y_CHUNKS);
            let mut anchored = false;
            generator.for_each_platform_instance_in_level_range(
                key_bounds,
                min_level,
                max_level,
                |platform| {
                    let y_offset = platform_level_chunk_y_offset(platform.level);
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
}
