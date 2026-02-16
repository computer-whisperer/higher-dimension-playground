use crate::shared::voxel::{BaseWorldKind, Chunk, ChunkPos, VoxelType, VoxelWorld, CHUNK_SIZE};
use crate::shared::worldfield::{Aabb4i, ChunkKey, ChunkPayload, RegionOverrideTree};

const FLAT_FLOOR_CHUNK_Y: i32 = -1;

#[derive(Debug)]
pub struct ServerWorldTree {
    base_kind: BaseWorldKind,
    flat_floor_chunk: Chunk,
    override_tree: RegionOverrideTree,
    world_dirty: bool,
}

impl ServerWorldTree {
    pub fn from_legacy_world(world: VoxelWorld) -> Self {
        let base_kind = world.base_kind();
        let flat_floor_chunk = match base_kind {
            BaseWorldKind::FlatFloor { material } => build_flat_floor_chunk(material),
            BaseWorldKind::Empty => Chunk::new(),
        };
        let override_tree =
            RegionOverrideTree::from_chunk_overrides(world.chunks.iter().map(|(&pos, chunk)| {
                (
                    ChunkKey::from_chunk_pos(pos),
                    ChunkPayload::from_chunk_compact(chunk),
                )
            }));
        Self {
            base_kind,
            flat_floor_chunk,
            override_tree,
            world_dirty: world.any_dirty(),
        }
    }

    pub fn to_legacy_world(&self) -> VoxelWorld {
        let mut world = VoxelWorld::new_with_base(self.base_kind);
        for (key, payload) in self.override_tree.collect_chunk_overrides() {
            if let Ok(chunk) = payload.to_voxel_chunk() {
                world.insert_chunk(key.to_chunk_pos(), chunk);
            }
        }
        world.clear_dirty();
        let _ = world.drain_pending_chunk_updates();
        world
    }

    pub fn base_kind(&self) -> BaseWorldKind {
        self.base_kind
    }

    pub fn override_tree(&self) -> &RegionOverrideTree {
        &self.override_tree
    }

    pub fn clone_base_chunk_or_empty_at(&self, pos: ChunkPos) -> Chunk {
        match self.base_kind {
            BaseWorldKind::Empty => Chunk::new(),
            BaseWorldKind::FlatFloor { .. } if pos.y == FLAT_FLOOR_CHUNK_Y => {
                self.flat_floor_chunk.clone()
            }
            BaseWorldKind::FlatFloor { .. } => Chunk::new(),
        }
    }

    pub fn base_chunk_y_bounds_for_scale(&self, chunk_scale: i32) -> Option<(i32, i32)> {
        let scale = chunk_scale.max(1);
        match self.base_kind {
            BaseWorldKind::Empty => None,
            BaseWorldKind::FlatFloor { .. } => {
                if self.flat_floor_chunk.is_empty() {
                    None
                } else {
                    let y = FLAT_FLOOR_CHUNK_Y.div_euclid(scale);
                    Some((y, y))
                }
            }
        }
    }

    pub fn has_chunk_override(&self, chunk_pos: ChunkPos) -> bool {
        self.override_tree
            .has_chunk_override(ChunkKey::from_chunk_pos(chunk_pos))
    }

    pub fn override_chunk_at(&self, chunk_pos: ChunkPos) -> Option<Chunk> {
        self.override_tree
            .chunk_override_payload(ChunkKey::from_chunk_pos(chunk_pos))
            .and_then(|payload| payload.to_voxel_chunk().ok())
    }

    pub fn set_chunk_override(&mut self, chunk_pos: ChunkPos, chunk: Chunk) -> bool {
        let payload = ChunkPayload::from_chunk_compact(&chunk);
        let changed = self
            .override_tree
            .set_chunk_override(ChunkKey::from_chunk_pos(chunk_pos), Some(payload));
        if changed {
            self.world_dirty = true;
        }
        changed
    }

    pub fn remove_chunk_override(&mut self, chunk_pos: ChunkPos) -> bool {
        let changed = self
            .override_tree
            .remove_chunk_override(ChunkKey::from_chunk_pos(chunk_pos));
        if changed {
            self.world_dirty = true;
        }
        changed
    }

    pub fn any_non_empty_override_in_bounds(&self, bounds: Aabb4i) -> bool {
        self.override_tree.any_non_empty_override_in_bounds(bounds)
    }

    pub fn collect_override_chunks(&self) -> Vec<(ChunkPos, Chunk)> {
        self.override_tree
            .collect_chunk_overrides()
            .into_iter()
            .filter_map(|(key, payload)| {
                payload
                    .to_voxel_chunk()
                    .ok()
                    .map(|chunk| (key.to_chunk_pos(), chunk))
            })
            .collect()
    }

    pub fn non_empty_chunk_count(&self) -> usize {
        self.override_tree.non_empty_chunk_count()
    }

    pub fn any_dirty(&self) -> bool {
        self.world_dirty
    }

    pub fn clear_dirty(&mut self) {
        self.world_dirty = false;
    }
}

fn build_flat_floor_chunk(material: VoxelType) -> Chunk {
    let mut chunk = Chunk::new();
    if material.is_air() {
        chunk.dirty = false;
        return chunk;
    }

    let local_y_top = CHUNK_SIZE - 1;
    let local_y_bottom = CHUNK_SIZE - 2;
    for x in 0..CHUNK_SIZE {
        for z in 0..CHUNK_SIZE {
            for w in 0..CHUNK_SIZE {
                chunk.set(x, local_y_top, z, w, material);
                chunk.set(x, local_y_bottom, z, w, material);
            }
        }
    }
    chunk.dirty = false;
    chunk
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn legacy_roundtrip_preserves_explicit_empty_override() {
        let mut world = VoxelWorld::new_with_base(BaseWorldKind::FlatFloor {
            material: VoxelType(6),
        });
        let floor = ChunkPos::new(0, -1, 0, 0);
        world.insert_chunk(floor, Chunk::new());

        let tree = ServerWorldTree::from_legacy_world(world);
        let rebuilt = tree.to_legacy_world();
        let override_chunk = rebuilt
            .override_chunk_at(floor)
            .expect("empty override exists");
        assert!(override_chunk.is_empty());
    }

    #[test]
    fn non_empty_chunk_count_tracks_tree_content() {
        let world = VoxelWorld::new_with_base(BaseWorldKind::Empty);
        let mut tree = ServerWorldTree::from_legacy_world(world);
        assert_eq!(tree.non_empty_chunk_count(), 0);

        let mut chunk = Chunk::new();
        chunk.set(0, 0, 0, 0, VoxelType(4));
        assert!(tree.set_chunk_override(ChunkPos::new(1, 0, 0, 0), chunk));
        assert_eq!(tree.non_empty_chunk_count(), 1);
        assert!(tree.remove_chunk_override(ChunkPos::new(1, 0, 0, 0)));
        assert_eq!(tree.non_empty_chunk_count(), 0);
    }
}
