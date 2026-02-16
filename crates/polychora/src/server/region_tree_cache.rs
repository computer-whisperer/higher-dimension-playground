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
        let old_chunks = self
            .interest_bounds
            .map(|old_bounds| collect_non_empty_chunks_in_bounds(&self.tree, old_bounds))
            .unwrap_or_default();
        let desired_chunks = collect_non_empty_chunks_from_core_in_bounds(core, bounds);
        apply_chunk_diff(&mut self.tree, &old_chunks, &desired_chunks);
        self.interest_bounds = Some(bounds);

        let load_positions = desired_chunks
            .iter()
            .filter_map(|(chunk_pos, payload)| match old_chunks.get(chunk_pos) {
                Some(old_payload) if old_payload == payload => None,
                _ => Some(*chunk_pos),
            })
            .collect();

        let unload_positions = old_chunks
            .keys()
            .filter_map(|chunk_pos| (!desired_chunks.contains_key(chunk_pos)).then_some(*chunk_pos))
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

fn collect_non_empty_region_chunks_from_core(
    node: &RegionTreeCore,
    out: &mut Vec<(ChunkPos, ChunkPayload)>,
) {
    match &node.kind {
        RegionNodeKind::Empty | RegionNodeKind::ProceduralRef(_) => {}
        RegionNodeKind::Uniform(material) => {
            if *material == 0 {
                return;
            }
            let payload = ChunkPayload::Uniform(*material);
            for chunk_w in node.bounds.min[3]..=node.bounds.max[3] {
                for chunk_z in node.bounds.min[2]..=node.bounds.max[2] {
                    for chunk_y in node.bounds.min[1]..=node.bounds.max[1] {
                        for chunk_x in node.bounds.min[0]..=node.bounds.max[0] {
                            out.push((
                                ChunkPos::new(chunk_x, chunk_y, chunk_z, chunk_w),
                                payload.clone(),
                            ));
                        }
                    }
                }
            }
        }
        RegionNodeKind::ChunkArray(chunk_array) => {
            let Ok(indices) = chunk_array.decode_dense_indices() else {
                return;
            };
            let Some(extents) = chunk_array.bounds.chunk_extents() else {
                return;
            };
            for chunk_w in chunk_array.bounds.min[3]..=chunk_array.bounds.max[3] {
                for chunk_z in chunk_array.bounds.min[2]..=chunk_array.bounds.max[2] {
                    for chunk_y in chunk_array.bounds.min[1]..=chunk_array.bounds.max[1] {
                        for chunk_x in chunk_array.bounds.min[0]..=chunk_array.bounds.max[0] {
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
                            out.push((
                                ChunkPos::new(chunk_x, chunk_y, chunk_z, chunk_w),
                                payload.clone(),
                            ));
                        }
                    }
                }
            }
        }
        RegionNodeKind::Branch(children) => {
            for child in children {
                collect_non_empty_region_chunks_from_core(child, out);
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

    let mut desired_pairs = Vec::new();
    collect_non_empty_region_chunks_from_core(core, &mut desired_pairs);
    desired_pairs
        .into_iter()
        .filter(|(chunk_pos, _)| {
            bounds.contains_chunk([chunk_pos.x, chunk_pos.y, chunk_pos.z, chunk_pos.w])
        })
        .collect()
}

fn collect_non_empty_chunks_in_bounds(
    tree: &RegionChunkTree,
    bounds: Aabb4i,
) -> HashMap<ChunkPos, ChunkPayload> {
    if !bounds.is_valid() {
        return HashMap::new();
    }
    tree.collect_chunks_in_bounds(bounds)
        .into_iter()
        .filter_map(|(key, payload)| {
            if !chunk_payload_has_content(&payload) {
                return None;
            }
            Some((key.to_chunk_pos(), payload))
        })
        .collect()
}

fn apply_chunk_diff(
    tree: &mut RegionChunkTree,
    old_chunks: &HashMap<ChunkPos, ChunkPayload>,
    desired_chunks: &HashMap<ChunkPos, ChunkPayload>,
) {
    for chunk_pos in old_chunks.keys() {
        if !desired_chunks.contains_key(chunk_pos) {
            let _ = tree.remove_chunk(ChunkKey::from_chunk_pos(*chunk_pos));
        }
    }

    for (chunk_pos, payload) in desired_chunks {
        if old_chunks.get(chunk_pos) == Some(payload) {
            continue;
        }
        let _ = tree.set_chunk(ChunkKey::from_chunk_pos(*chunk_pos), Some(payload.clone()));
    }
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
