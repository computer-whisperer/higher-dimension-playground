use crate::shared::voxel::ChunkPos;
use crate::shared::worldfield::{Aabb4i, ChunkKey, ChunkPayload, RegionChunkTree, RegionTreeCore};

#[derive(Clone, Debug, Default)]
pub struct RegionTreeWorkingSet {
    tree: RegionChunkTree,
    interest_bounds: Option<Aabb4i>,
}

#[derive(Debug)]
pub struct RegionTreeRefreshResult {
    pub load_chunks: Vec<(ChunkPos, ChunkPayload)>,
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
        let patch_bounds = self
            .interest_bounds
            .map(|old_bounds| union_aabb(old_bounds, bounds))
            .unwrap_or(bounds);
        let patch = self.tree.apply_non_empty_core_in_bounds(patch_bounds, core);
        self.interest_bounds = Some(bounds);

        let load_chunks = patch
            .upserts
            .iter()
            .map(|(key, payload)| (key.to_chunk_pos(), payload.clone()))
            .collect();

        let unload_positions = patch
            .removals
            .iter()
            .map(|key| key.to_chunk_pos())
            .collect();

        RegionTreeRefreshResult {
            load_chunks,
            unload_positions,
        }
    }
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
        assert_eq!(
            first.load_chunks,
            vec![(ChunkPos::new(0, 0, 0, 0), ChunkPayload::Uniform(3))]
        );
        assert!(first.unload_positions.is_empty());
        assert!(working.contains_chunk(ChunkPos::new(0, 0, 0, 0)));

        let second = working.refresh_from_core(bounds, &core);
        assert!(second.load_chunks.is_empty());
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
        assert!(diff.load_chunks.is_empty());
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
        assert_eq!(
            shifted.load_chunks,
            vec![(ChunkPos::new(1, 0, 0, 0), ChunkPayload::Uniform(5))]
        );
        assert!(!working.contains_chunk(ChunkPos::new(0, 0, 0, 0)));
        assert!(working.contains_chunk(ChunkPos::new(1, 0, 0, 0)));
    }

    #[test]
    fn refresh_reports_load_when_payload_changes_in_place() {
        let bounds = one_chunk_bounds();
        let mut working = RegionTreeWorkingSet::new();
        let first_core = RegionTreeCore {
            bounds,
            kind: RegionNodeKind::Uniform(2),
            generator_version_hash: 0,
        };
        let second_core = RegionTreeCore {
            bounds,
            kind: RegionNodeKind::Uniform(8),
            generator_version_hash: 0,
        };

        let _ = working.refresh_from_core(bounds, &first_core);
        let updated = working.refresh_from_core(bounds, &second_core);
        assert_eq!(
            updated.load_chunks,
            vec![(ChunkPos::new(0, 0, 0, 0), ChunkPayload::Uniform(8))]
        );
        assert!(updated.unload_positions.is_empty());
    }
}
