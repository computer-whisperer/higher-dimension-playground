use crate::shared::voxel::ChunkPos;
use crate::shared::worldfield::{Aabb4i, ChunkKey, RegionChunkTree, RegionTreeCore};
use std::sync::OnceLock;
use std::time::Instant;

#[derive(Clone, Debug, Default)]
pub struct RegionTreeWorkingSet {
    tree: RegionChunkTree,
    interest_bounds: Option<Aabb4i>,
}

#[derive(Debug)]
pub struct RegionTreeRefreshResult {
    pub changed_bounds: Option<Aabb4i>,
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
        let merge_start = Instant::now();
        let changed_bounds = self
            .tree
            .splice_non_empty_core_in_bounds(patch_bounds, core);
        if self.interest_bounds != Some(bounds) {
            let _ = self
                .tree
                .lazy_drop_outside_bounds(bounds, region_tree_lazy_drop_budget());
        }
        let merge_elapsed = merge_start.elapsed();
        let merge_elapsed_ms = merge_elapsed.as_secs_f64() * 1000.0;
        if region_tree_merge_diag_enabled() || merge_elapsed_ms >= 100.0 {
            eprintln!(
                "[region-tree-merge] elapsed_ms={:.3} query={:?}->{:?} patch={:?}->{:?} keep={:?}->{:?} core={:?}->{:?} patch_cells={} changed={} changed_bounds={:?} lazy_drop_budget={}",
                merge_elapsed_ms,
                bounds.min,
                bounds.max,
                patch_bounds.min,
                patch_bounds.max,
                bounds.min,
                bounds.max,
                core.bounds.min,
                core.bounds.max,
                patch_bounds.chunk_cell_count().unwrap_or(0),
                changed_bounds.is_some(),
                changed_bounds,
                region_tree_lazy_drop_budget()
            );
        }
        self.interest_bounds = Some(bounds);

        RegionTreeRefreshResult { changed_bounds }
    }
}

fn region_tree_merge_diag_enabled() -> bool {
    static ENABLED: OnceLock<bool> = OnceLock::new();
    *ENABLED.get_or_init(|| {
        std::env::var("R4D_REGION_TREE_MERGE_DIAG")
            .ok()
            .map(|value| {
                let normalized = value.trim().to_ascii_lowercase();
                matches!(normalized.as_str(), "1" | "true" | "yes" | "on")
            })
            .unwrap_or(false)
    })
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

fn region_tree_lazy_drop_budget() -> usize {
    static BUDGET: OnceLock<usize> = OnceLock::new();
    *BUDGET.get_or_init(|| {
        std::env::var("R4D_REGION_TREE_LAZY_DROP_BUDGET")
            .ok()
            .and_then(|value| value.trim().parse::<usize>().ok())
            .unwrap_or(8)
    })
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
        assert_eq!(first.changed_bounds, Some(bounds));
        assert!(working.contains_chunk(ChunkPos::new(0, 0, 0, 0)));

        let second = working.refresh_from_core(bounds, &core);
        assert!(second.changed_bounds.is_none());
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
        assert_eq!(diff.changed_bounds, Some(bounds));
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

        assert_eq!(
            shifted.changed_bounds,
            Some(Aabb4i::new([0, 0, 0, 0], [1, 0, 0, 0]))
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
        assert_eq!(updated.changed_bounds, Some(bounds));
    }
}
