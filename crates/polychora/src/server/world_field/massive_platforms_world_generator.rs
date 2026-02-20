use super::{QueryDetail, QueryVolume, WorldField};
use crate::server::procgen;
use crate::shared::chunk_payload::ChunkPayload;
use crate::shared::region_tree::{
    chunk_key_from_chunk_pos, RegionChunkTree, RegionNodeKind, RegionTreeCore,
};
use crate::shared::spatial::Aabb4i;
use crate::shared::voxel::{BaseWorldKind, VoxelType, CHUNK_VOLUME};
use std::collections::HashSet;
use std::sync::Arc;

type DenseChunk = [VoxelType; CHUNK_VOLUME];

const PLATFORM_GRID_STRIDE_XZW_CHUNKS: i32 = 48;
const PLATFORM_LEVEL_STRIDE_Y_CHUNKS: i32 = 12;
const PLATFORM_THICKNESS_CHUNKS: i32 = 3;
const PLATFORM_HALF_EXTENT_MIN_CHUNKS: i32 = 12;
const PLATFORM_HALF_EXTENT_MAX_CHUNKS: i32 = 20;
const PLATFORM_CENTER_JITTER_CHUNKS: i32 = 8;
const PLATFORM_PRESENCE_PERCENT: u64 = 86;

const PLATFORM_HASH_SALT_PRESENCE: u64 = 0x91f4_2b7e_87f1_3a55;
const PLATFORM_HASH_SALT_EXTENT_X: u64 = 0x58d7_6c29_2a0b_4cd1;
const PLATFORM_HASH_SALT_EXTENT_Z: u64 = 0x7ef8_18a3_02e5_2b3f;
const PLATFORM_HASH_SALT_EXTENT_W: u64 = 0x4cc2_743e_91ad_0627;
const PLATFORM_HASH_SALT_JITTER_X: u64 = 0x0a1e_d952_75c4_6619;
const PLATFORM_HASH_SALT_JITTER_Z: u64 = 0x8f63_d4b2_3c14_20e3;
const PLATFORM_HASH_SALT_JITTER_W: u64 = 0x3ce7_b890_5d61_489f;

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
        let min_y = level * PLATFORM_LEVEL_STRIDE_Y_CHUNKS;
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

        let min_level = (bounds.min[1] - (PLATFORM_THICKNESS_CHUNKS - 1))
            .div_euclid(PLATFORM_LEVEL_STRIDE_Y_CHUNKS);
        let max_level = bounds.max[1].div_euclid(PLATFORM_LEVEL_STRIDE_Y_CHUNKS);

        for level in min_level..=max_level {
            for cell_w in min_cell_w..=max_cell_w {
                for cell_z in min_cell_z..=max_cell_z {
                    for cell_x in min_cell_x..=max_cell_x {
                        let Some(platform_bounds) =
                            self.sample_platform_bounds(level, cell_x, cell_z, cell_w)
                        else {
                            continue;
                        };
                        let Some(intersection) = intersect_bounds(bounds, platform_bounds) else {
                            continue;
                        };
                        visitor(intersection);
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
        let candidates = procgen::structure_chunk_positions_for_bounds_with_keepout(
            self.world_seed,
            bounds,
            self.procgen_keepout_cells(),
        );
        for chunk_pos in candidates {
            let Some(structure_chunk) = procgen::generate_structure_chunk_with_keepout(
                self.world_seed,
                chunk_pos,
                self.procgen_keepout_cells(),
            ) else {
                continue;
            };
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
}
