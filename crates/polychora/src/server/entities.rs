use crate::shared::protocol::{EntityKind, EntitySnapshot};
use std::collections::HashMap;

#[derive(Clone, Debug)]
pub struct EntityState {
    pub entity_id: u64,
    pub kind: EntityKind,
    pub position: [f32; 4],
    pub orientation: [f32; 4],
    pub scale: f32,
    pub material: u8,
    pub last_update_ms: u64,
}

impl EntityState {
    fn snapshot(&self) -> EntitySnapshot {
        EntitySnapshot {
            entity_id: self.entity_id,
            kind: self.kind,
            position: self.position,
            orientation: self.orientation,
            scale: self.scale,
            material: self.material,
            last_update_ms: self.last_update_ms,
        }
    }
}

#[derive(Debug)]
pub struct EntityStore {
    next_entity_id: u64,
    entities: HashMap<u64, EntityState>,
}

impl EntityStore {
    pub fn new() -> Self {
        Self {
            next_entity_id: 1,
            entities: HashMap::new(),
        }
    }

    pub fn len(&self) -> usize {
        self.entities.len()
    }

    pub fn is_empty(&self) -> bool {
        self.entities.is_empty()
    }

    pub fn snapshots(&self) -> Vec<EntitySnapshot> {
        self.entities.values().map(EntityState::snapshot).collect()
    }

    pub fn sorted_snapshots(&self) -> Vec<EntitySnapshot> {
        let mut all = self.snapshots();
        all.sort_by_key(|e| e.entity_id);
        all
    }

    pub fn spawn(
        &mut self,
        kind: EntityKind,
        position: [f32; 4],
        orientation: [f32; 4],
        scale: f32,
        material: u8,
        last_update_ms: u64,
    ) -> u64 {
        let entity_id = self.next_entity_id;
        self.next_entity_id = self.next_entity_id.wrapping_add(1).max(1);
        let entity = EntityState {
            entity_id,
            kind,
            position,
            orientation,
            scale,
            material,
            last_update_ms,
        };
        self.entities.insert(entity_id, entity);
        entity_id
    }
}
