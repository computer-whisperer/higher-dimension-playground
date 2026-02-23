use crate::shared::entity_types::{
    self, EntityCategory, ENTITY_TEST_CUBE, ENTITY_TEST_DRIFTER, ENTITY_TEST_ROTOR,
};
use crate::shared::protocol::{Entity, EntitySnapshot};
use std::collections::HashMap;

use super::normalize4_or_default;

pub type EntityId = u64;

pub fn update_entity_motion(
    entity: &mut Entity,
    position: [f32; 4],
    orientation: [f32; 4],
    now_ms: u64,
    last_update_ms: &mut u64,
) {
    let dt_ms = now_ms.saturating_sub(*last_update_ms);
    if dt_ms > 0 {
        let inv_dt_s = 1000.0 / dt_ms as f32;
        entity.pose.velocity = [
            (position[0] - entity.pose.position[0]) * inv_dt_s,
            (position[1] - entity.pose.position[1]) * inv_dt_s,
            (position[2] - entity.pose.position[2]) * inv_dt_s,
            (position[3] - entity.pose.position[3]) * inv_dt_s,
        ];
    }
    entity.pose.position = position;
    entity.pose.orientation = orientation;
    *last_update_ms = now_ms;
}

#[derive(Clone, Debug)]
pub struct EntityState {
    pub entity_id: EntityId,
    pub entity: Entity,
    pub last_update_ms: u64,
    // Server-only bookkeeping (not on wire)
    pub home_position: [f32; 4],
    pub base_scale: f32,
}

impl EntityState {
    fn category(&self) -> EntityCategory {
        entity_types::category_for(self.entity.namespace, self.entity.entity_type)
    }

    fn type_key(&self) -> (u32, u32) {
        (self.entity.namespace, self.entity.entity_type)
    }

    fn simulate(&mut self, now_ms: u64) -> bool {
        if self.category() != EntityCategory::Accent {
            return false;
        }
        let previous_position = self.entity.pose.position;
        let previous_orientation = self.entity.pose.orientation;
        let previous_scale = self.entity.pose.scale;
        let t = now_ms as f32 * 0.001;
        let type_key = self.type_key();
        let (next_position, next_orientation, next_scale) = if type_key == ENTITY_TEST_CUBE {
            let phase_a = t * 0.65 + self.entity_id as f32 * 0.61;
            let phase_b = t * 0.41 + self.entity_id as f32 * 0.23;
            (
                [
                    self.home_position[0] + 0.18 * phase_b.cos(),
                    self.home_position[1] + 0.26 * phase_a.sin(),
                    self.home_position[2] + 0.18 * phase_a.cos(),
                    self.home_position[3] + 0.18 * phase_b.sin(),
                ],
                normalize4_or_default(
                    [
                        phase_a.cos(),
                        0.35 * phase_b.cos(),
                        phase_a.sin(),
                        0.70 * phase_b.sin(),
                    ],
                    [0.0, 0.0, 1.0, 0.0],
                ),
                self.base_scale * (1.0 + 0.06 * (phase_a * 0.7).sin()),
            )
        } else if type_key == ENTITY_TEST_ROTOR {
            let phase = t * 0.95 + self.entity_id as f32 * 0.41;
            let wobble = t * 0.57 + self.entity_id as f32 * 0.19;
            (
                [
                    self.home_position[0] + 0.36 * phase.cos(),
                    self.home_position[1] + 0.10 * wobble.sin(),
                    self.home_position[2] + 0.36 * phase.sin(),
                    self.home_position[3] + 0.24 * (phase * 1.3).cos(),
                ],
                normalize4_or_default(
                    [
                        -phase.sin(),
                        0.25 * wobble.cos(),
                        phase.cos(),
                        -0.80 * (phase * 1.3).sin(),
                    ],
                    [0.0, 0.0, 1.0, 0.0],
                ),
                self.base_scale * (1.0 + 0.10 * (wobble * 1.2).sin()),
            )
        } else if type_key == ENTITY_TEST_DRIFTER {
            let phase_a = t * 0.33 + self.entity_id as f32 * 0.77;
            let phase_b = t * 0.53 + self.entity_id as f32 * 0.29;
            (
                [
                    self.home_position[0] + 0.46 * phase_a.sin(),
                    self.home_position[1] + 0.18 * phase_b.cos(),
                    self.home_position[2] + 0.34 * (phase_a * 1.4).cos(),
                    self.home_position[3] + 0.42 * phase_b.sin(),
                ],
                normalize4_or_default(
                    [
                        (phase_a * 1.4).sin(),
                        -0.35 * phase_b.sin(),
                        -(phase_a * 1.4).cos(),
                        0.90 * phase_b.cos(),
                    ],
                    [0.0, 0.0, 1.0, 0.0],
                ),
                self.base_scale * (1.0 + 0.08 * (phase_a + phase_b).sin()),
            )
        } else {
            // Unknown accent or mob/player — no server-side animation
            (
                self.entity.pose.position,
                self.entity.pose.orientation,
                self.base_scale.max(0.01),
            )
        };
        self.entity.pose.scale = next_scale;
        update_entity_motion(
            &mut self.entity,
            next_position,
            next_orientation,
            now_ms,
            &mut self.last_update_ms,
        );
        previous_position != next_position
            || previous_orientation != next_orientation
            || previous_scale != next_scale
    }

    pub fn snapshot(&self) -> EntitySnapshot {
        EntitySnapshot {
            entity_id: self.entity_id,
            entity: self.entity.clone(),
            owner_client_id: None,
            display_name: None,
            last_update_ms: self.last_update_ms,
        }
    }
}

#[derive(Debug)]
pub struct EntityStore {
    entities: HashMap<EntityId, EntityState>,
}

impl EntityStore {
    pub fn new() -> Self {
        Self {
            entities: HashMap::new(),
        }
    }

    pub fn len(&self) -> usize {
        self.entities.len()
    }

    pub fn snapshot(&self, entity_id: EntityId) -> Option<EntitySnapshot> {
        self.entities.get(&entity_id).map(EntityState::snapshot)
    }

    pub fn despawn(&mut self, entity_id: EntityId) -> bool {
        self.entities.remove(&entity_id).is_some()
    }

    pub fn simulate(&mut self, now_ms: u64) -> Vec<EntityId> {
        let mut moved = Vec::new();
        for entity in self.entities.values_mut() {
            if entity.simulate(now_ms) {
                moved.push(entity.entity_id);
            }
        }
        moved
    }

    pub fn set_motion_state(
        &mut self,
        entity_id: EntityId,
        position: [f32; 4],
        orientation: [f32; 4],
        now_ms: u64,
    ) -> bool {
        let Some(entity) = self.entities.get_mut(&entity_id) else {
            return false;
        };
        update_entity_motion(
            &mut entity.entity,
            position,
            normalize4_or_default(orientation, [0.0, 0.0, 1.0, 0.0]),
            now_ms,
            &mut entity.last_update_ms,
        );
        true
    }

    pub fn spawn(
        &mut self,
        entity_id: EntityId,
        entity: Entity,
        last_update_ms: u64,
    ) -> EntityId {
        let home_position = entity.pose.position;
        let base_scale = entity.pose.scale;
        let state = EntityState {
            entity_id,
            entity,
            last_update_ms,
            home_position,
            base_scale,
        };
        let prior = self.entities.insert(entity_id, state);
        assert!(prior.is_none(), "duplicate entity id: {entity_id}");
        entity_id
    }
}
