use crate::shared::protocol::{EntityKind, EntitySnapshot};
use std::collections::HashMap;

fn normalize4_with_fallback(v: [f32; 4], fallback: [f32; 4]) -> [f32; 4] {
    let len_sq = v[0] * v[0] + v[1] * v[1] + v[2] * v[2] + v[3] * v[3];
    if len_sq <= 1e-8 {
        return fallback;
    }
    let inv_len = len_sq.sqrt().recip();
    [
        v[0] * inv_len,
        v[1] * inv_len,
        v[2] * inv_len,
        v[3] * inv_len,
    ]
}

#[derive(Clone, Debug)]
pub struct EntityState {
    pub entity_id: u64,
    pub kind: EntityKind,
    home_position: [f32; 4],
    base_scale: f32,
    pub position: [f32; 4],
    pub orientation: [f32; 4],
    pub scale: f32,
    pub material: u8,
    pub last_update_ms: u64,
}

impl EntityState {
    fn simulate(&mut self, now_ms: u64) {
        let t = now_ms as f32 * 0.001;
        match self.kind {
            EntityKind::TestCube => {
                // Keep test entities moving in 4D using two coupled phase loops.
                let phase_a = t * 0.65 + self.entity_id as f32 * 0.61;
                let phase_b = t * 0.41 + self.entity_id as f32 * 0.23;
                self.position = [
                    self.home_position[0] + 0.18 * phase_b.cos(),
                    self.home_position[1] + 0.26 * phase_a.sin(),
                    self.home_position[2] + 0.18 * phase_a.cos(),
                    self.home_position[3] + 0.18 * phase_b.sin(),
                ];
                self.orientation = normalize4_with_fallback(
                    [
                        phase_a.cos(),
                        0.35 * phase_b.cos(),
                        phase_a.sin(),
                        0.70 * phase_b.sin(),
                    ],
                    [0.0, 0.0, 1.0, 0.0],
                );
                self.scale = self.base_scale * (1.0 + 0.06 * (phase_a * 0.7).sin());
            }
            EntityKind::TestRotor => {
                let phase = t * 0.95 + self.entity_id as f32 * 0.41;
                let wobble = t * 0.57 + self.entity_id as f32 * 0.19;
                self.position = [
                    self.home_position[0] + 0.36 * phase.cos(),
                    self.home_position[1] + 0.10 * wobble.sin(),
                    self.home_position[2] + 0.36 * phase.sin(),
                    self.home_position[3] + 0.24 * (phase * 1.3).cos(),
                ];
                // Tangent-like forward in XZW with a mild Y wobble for 4D feel.
                self.orientation = normalize4_with_fallback(
                    [
                        -phase.sin(),
                        0.25 * wobble.cos(),
                        phase.cos(),
                        -0.80 * (phase * 1.3).sin(),
                    ],
                    [0.0, 0.0, 1.0, 0.0],
                );
                self.scale = self.base_scale * (1.0 + 0.10 * (wobble * 1.2).sin());
            }
            EntityKind::TestDrifter => {
                let phase_a = t * 0.33 + self.entity_id as f32 * 0.77;
                let phase_b = t * 0.53 + self.entity_id as f32 * 0.29;
                self.position = [
                    self.home_position[0] + 0.46 * phase_a.sin(),
                    self.home_position[1] + 0.18 * phase_b.cos(),
                    self.home_position[2] + 0.34 * (phase_a * 1.4).cos(),
                    self.home_position[3] + 0.42 * phase_b.sin(),
                ];
                self.orientation = normalize4_with_fallback(
                    [
                        (phase_a * 1.4).sin(),
                        -0.35 * phase_b.sin(),
                        -(phase_a * 1.4).cos(),
                        0.90 * phase_b.cos(),
                    ],
                    [0.0, 0.0, 1.0, 0.0],
                );
                self.scale = self.base_scale * (1.0 + 0.08 * (phase_a + phase_b).sin());
            }
        }
        self.last_update_ms = now_ms;
    }

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

    pub fn simulate(&mut self, now_ms: u64) {
        for entity in self.entities.values_mut() {
            entity.simulate(now_ms);
        }
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
            home_position: position,
            base_scale: scale,
            position,
            orientation: normalize4_with_fallback(orientation, [0.0, 0.0, 1.0, 0.0]),
            scale,
            material,
            last_update_ms,
        };
        self.entities.insert(entity_id, entity);
        entity_id
    }
}
