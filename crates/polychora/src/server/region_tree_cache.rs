use crate::shared::voxel::ChunkPos;
use crate::shared::worldfield::{
    Aabb4i, ChunkKey, ChunkPayload, RegionChunkTree, RegionNodeKind, RegionTreeCore,
};
use std::collections::HashMap;

#[derive(Clone, Debug, Default)]
pub struct RegionTreeWorkingSet {
    tree: RegionChunkTree,
    interest_bounds: Option<Aabb4i>,
}

#[derive(Debug)]
pub struct RegionTreeRefreshResult {
    pub desired_chunks: HashMap<ChunkPos, ChunkPayload>,
    pub load_positions: Vec<ChunkPos>,
    pub unload_positions: Vec<ChunkPos>,
}

impl RegionTreeWorkingSet {
    pub fn new() -> Self {
        Self::default()
    }

    pub fn contains_chunk(&self, chunk_pos: ChunkPos) -> bool {
        self.tree.has_chunk(ChunkKey::from_chunk_pos(chunk_pos))
    }

    pub fn remove_chunk(&mut self, chunk_pos: ChunkPos) -> bool {
        self.tree.remove_chunk(ChunkKey::from_chunk_pos(chunk_pos))
    }

    pub fn refresh_from_core(
        &mut self,
        bounds: Aabb4i,
        core: &RegionTreeCore,
    ) -> RegionTreeRefreshResult {
        let desired_chunks = collect_non_empty_chunks_from_core_in_bounds(core, bounds);
        let patch_bounds = self
            .interest_bounds
            .map(|old_bounds| union_aabb(old_bounds, bounds))
            .unwrap_or(bounds);
        let patch = self.tree.diff_chunks_in_bounds(
            patch_bounds,
            desired_chunks.iter().map(|(chunk_pos, payload)| {
                (ChunkKey::from_chunk_pos(*chunk_pos), payload.clone())
            }),
        );
        self.tree.apply_chunk_diff(&patch);
        self.interest_bounds = Some(bounds);

        let load_positions = patch
            .upserts
            .iter()
            .map(|(key, _)| key.to_chunk_pos())
            .collect();

        let unload_positions = patch
            .removals
            .iter()
            .map(|key| key.to_chunk_pos())
            .collect();

        RegionTreeRefreshResult {
            desired_chunks,
            load_positions,
            unload_positions,
        }
    }
}

fn chunk_payload_has_content(payload: &ChunkPayload) -> bool {
    match payload {
        ChunkPayload::Empty => false,
        ChunkPayload::Uniform(material) => *material != 0,
        ChunkPayload::Dense16 { materials } => materials.iter().any(|material| *material != 0),
        ChunkPayload::PalettePacked { .. } => payload
            .dense_materials()
            .map(|materials| materials.into_iter().any(|material| material != 0))
            .unwrap_or(true),
    }
}

fn linear_chunk_cell_index(coords: [usize; 4], dims: [usize; 4]) -> usize {
    coords[0] + dims[0] * (coords[1] + dims[1] * (coords[2] + dims[2] * coords[3]))
}

fn collect_non_empty_region_chunks_from_core_in_bounds(
    node: &RegionTreeCore,
    query_bounds: Aabb4i,
    out: &mut HashMap<ChunkPos, ChunkPayload>,
) {
    let Some(node_intersection) = intersect_aabb(node.bounds, query_bounds) else {
        return;
    };

    match &node.kind {
        RegionNodeKind::Empty | RegionNodeKind::ProceduralRef(_) => {}
        RegionNodeKind::Uniform(material) => {
            if *material == 0 {
                return;
            }
            let payload = ChunkPayload::Uniform(*material);
            for chunk_w in node_intersection.min[3]..=node_intersection.max[3] {
                for chunk_z in node_intersection.min[2]..=node_intersection.max[2] {
                    for chunk_y in node_intersection.min[1]..=node_intersection.max[1] {
                        for chunk_x in node_intersection.min[0]..=node_intersection.max[0] {
                            out.insert(
                                ChunkPos::new(chunk_x, chunk_y, chunk_z, chunk_w),
                                payload.clone(),
                            );
                        }
                    }
                }
            }
        }
        RegionNodeKind::ChunkArray(chunk_array) => {
            let Some(chunk_array_intersection) = intersect_aabb(chunk_array.bounds, query_bounds)
            else {
                return;
            };
            let Ok(indices) = chunk_array.decode_dense_indices() else {
                return;
            };
            let Some(extents) = chunk_array.bounds.chunk_extents() else {
                return;
            };
            for chunk_w in chunk_array_intersection.min[3]..=chunk_array_intersection.max[3] {
                for chunk_z in chunk_array_intersection.min[2]..=chunk_array_intersection.max[2] {
                    for chunk_y in chunk_array_intersection.min[1]..=chunk_array_intersection.max[1]
                    {
                        for chunk_x in
                            chunk_array_intersection.min[0]..=chunk_array_intersection.max[0]
                        {
                            let local = [
                                (chunk_x - chunk_array.bounds.min[0]) as usize,
                                (chunk_y - chunk_array.bounds.min[1]) as usize,
                                (chunk_z - chunk_array.bounds.min[2]) as usize,
                                (chunk_w - chunk_array.bounds.min[3]) as usize,
                            ];
                            let linear_idx = linear_chunk_cell_index(local, extents);
                            let Some(palette_idx) = indices.get(linear_idx) else {
                                continue;
                            };
                            let Some(payload) =
                                chunk_array.chunk_palette.get(*palette_idx as usize)
                            else {
                                continue;
                            };
                            if !chunk_payload_has_content(payload) {
                                continue;
                            }
                            out.insert(
                                ChunkPos::new(chunk_x, chunk_y, chunk_z, chunk_w),
                                payload.clone(),
                            );
                        }
                    }
                }
            }
        }
        RegionNodeKind::Branch(children) => {
            for child in children {
                collect_non_empty_region_chunks_from_core_in_bounds(child, query_bounds, out);
            }
        }
    }
}

fn collect_non_empty_chunks_from_core_in_bounds(
    core: &RegionTreeCore,
    bounds: Aabb4i,
) -> HashMap<ChunkPos, ChunkPayload> {
    if !bounds.is_valid() {
        return HashMap::new();
    }
    let mut out = HashMap::new();
    collect_non_empty_region_chunks_from_core_in_bounds(core, bounds, &mut out);
    out
}

fn union_aabb(a: Aabb4i, b: Aabb4i) -> Aabb4i {
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

fn intersect_aabb(a: Aabb4i, b: Aabb4i) -> Option<Aabb4i> {
    if !a.intersects(&b) {
        return None;
    }
    Some(Aabb4i::new(
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
    ))
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::shared::worldfield::{Aabb4i, RegionNodeKind, RegionTreeCore};

    fn one_chunk_bounds() -> Aabb4i {
        Aabb4i::new([0, 0, 0, 0], [0, 0, 0, 0])
    }

    #[test]
    fn refresh_reports_load_for_new_chunk_and_noop_for_identical_refresh() {
        let bounds = one_chunk_bounds();
        let mut working = RegionTreeWorkingSet::new();
        let core = RegionTreeCore {
            bounds,
            kind: RegionNodeKind::Uniform(3),
            generator_version_hash: 0,
        };

        let first = working.refresh_from_core(bounds, &core);
        assert_eq!(first.load_positions, vec![ChunkPos::new(0, 0, 0, 0)]);
        assert!(first.unload_positions.is_empty());
        assert!(working.contains_chunk(ChunkPos::new(0, 0, 0, 0)));

        let second = working.refresh_from_core(bounds, &core);
        assert!(second.load_positions.is_empty());
        assert!(second.unload_positions.is_empty());
    }

    #[test]
    fn refresh_reports_unload_when_chunk_becomes_empty() {
        let bounds = one_chunk_bounds();
        let mut working = RegionTreeWorkingSet::new();
        let filled = RegionTreeCore {
            bounds,
            kind: RegionNodeKind::Uniform(4),
            generator_version_hash: 0,
        };
        let empty = RegionTreeCore {
            bounds,
            kind: RegionNodeKind::Empty,
            generator_version_hash: 0,
        };

        let _ = working.refresh_from_core(bounds, &filled);
        let diff = working.refresh_from_core(bounds, &empty);
        assert!(diff.load_positions.is_empty());
        assert_eq!(diff.unload_positions, vec![ChunkPos::new(0, 0, 0, 0)]);
        assert!(!working.contains_chunk(ChunkPos::new(0, 0, 0, 0)));
    }

    #[test]
    fn refresh_window_shift_removes_stale_chunks_and_loads_new_chunks() {
        let mut working = RegionTreeWorkingSet::new();
        let left_bounds = Aabb4i::new([0, 0, 0, 0], [0, 0, 0, 0]);
        let right_bounds = Aabb4i::new([1, 0, 0, 0], [1, 0, 0, 0]);

        let left_core = RegionTreeCore {
            bounds: left_bounds,
            kind: RegionNodeKind::Uniform(2),
            generator_version_hash: 0,
        };
        let right_core = RegionTreeCore {
            bounds: right_bounds,
            kind: RegionNodeKind::Uniform(5),
            generator_version_hash: 0,
        };

        let _ = working.refresh_from_core(left_bounds, &left_core);
        let shifted = working.refresh_from_core(right_bounds, &right_core);

        assert_eq!(shifted.unload_positions, vec![ChunkPos::new(0, 0, 0, 0)]);
        assert_eq!(shifted.load_positions, vec![ChunkPos::new(1, 0, 0, 0)]);
        assert!(!working.contains_chunk(ChunkPos::new(0, 0, 0, 0)));
        assert!(working.contains_chunk(ChunkPos::new(1, 0, 0, 0)));
    }
}
