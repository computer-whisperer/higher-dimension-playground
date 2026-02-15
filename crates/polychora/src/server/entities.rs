use crate::shared::protocol::{EntityKind, EntitySnapshot};
use std::collections::HashMap;

pub type EntityId = u64;

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum EntityClass {
    Player,
    Accent,
    Mob,
}

const ENTITY_CLASSES: [EntityClass; 3] =
    [EntityClass::Player, EntityClass::Accent, EntityClass::Mob];

#[derive(Clone, Copy, Debug)]
pub struct EntityCore {
    pub entity_id: EntityId,
    pub class: EntityClass,
    pub position: [f32; 4],
    pub orientation: [f32; 4],
    pub velocity: [f32; 4],
    pub last_update_ms: u64,
}

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

pub fn update_core_motion(
    core: &mut EntityCore,
    position: [f32; 4],
    orientation: [f32; 4],
    now_ms: u64,
) {
    let dt_ms = now_ms.saturating_sub(core.last_update_ms);
    if dt_ms > 0 {
        let inv_dt_s = 1000.0 / dt_ms as f32;
        core.velocity = [
            (position[0] - core.position[0]) * inv_dt_s,
            (position[1] - core.position[1]) * inv_dt_s,
            (position[2] - core.position[2]) * inv_dt_s,
            (position[3] - core.position[3]) * inv_dt_s,
        ];
    }
    core.position = position;
    core.orientation = orientation;
    core.last_update_ms = now_ms;
}

#[derive(Clone, Debug)]
pub struct EntityState {
    pub core: EntityCore,
    pub kind: EntityKind,
    home_position: [f32; 4],
    base_scale: f32,
    pub scale: f32,
    pub material: u8,
}

impl EntityState {
    fn simulate(&mut self, now_ms: u64) {
        debug_assert!(ENTITY_CLASSES.contains(&self.core.class));
        let t = now_ms as f32 * 0.001;
        let (next_position, next_orientation, next_scale) = match self.kind {
            EntityKind::TestCube => {
                // Keep test entities moving in 4D using two coupled phase loops.
                let phase_a = t * 0.65 + self.core.entity_id as f32 * 0.61;
                let phase_b = t * 0.41 + self.core.entity_id as f32 * 0.23;
                (
                    [
                        self.home_position[0] + 0.18 * phase_b.cos(),
                        self.home_position[1] + 0.26 * phase_a.sin(),
                        self.home_position[2] + 0.18 * phase_a.cos(),
                        self.home_position[3] + 0.18 * phase_b.sin(),
                    ],
                    normalize4_with_fallback(
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
            }
            EntityKind::TestRotor => {
                let phase = t * 0.95 + self.core.entity_id as f32 * 0.41;
                let wobble = t * 0.57 + self.core.entity_id as f32 * 0.19;
                // Tangent-like forward in XZW with a mild Y wobble for 4D feel.
                (
                    [
                        self.home_position[0] + 0.36 * phase.cos(),
                        self.home_position[1] + 0.10 * wobble.sin(),
                        self.home_position[2] + 0.36 * phase.sin(),
                        self.home_position[3] + 0.24 * (phase * 1.3).cos(),
                    ],
                    normalize4_with_fallback(
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
            }
            EntityKind::TestDrifter => {
                let phase_a = t * 0.33 + self.core.entity_id as f32 * 0.77;
                let phase_b = t * 0.53 + self.core.entity_id as f32 * 0.29;
                (
                    [
                        self.home_position[0] + 0.46 * phase_a.sin(),
                        self.home_position[1] + 0.18 * phase_b.cos(),
                        self.home_position[2] + 0.34 * (phase_a * 1.4).cos(),
                        self.home_position[3] + 0.42 * phase_b.sin(),
                    ],
                    normalize4_with_fallback(
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
            }
        };
        self.scale = next_scale;
        update_core_motion(&mut self.core, next_position, next_orientation, now_ms);
    }

    fn snapshot(&self) -> EntitySnapshot {
        EntitySnapshot {
            entity_id: self.core.entity_id,
            kind: self.kind,
            position: self.core.position,
            orientation: self.core.orientation,
            velocity: self.core.velocity,
            scale: self.scale,
            material: self.material,
            last_update_ms: self.core.last_update_ms,
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

    pub fn is_empty(&self) -> bool {
        self.entities.is_empty()
    }

    pub fn snapshot(&self, entity_id: EntityId) -> Option<EntitySnapshot> {
        self.entities.get(&entity_id).map(EntityState::snapshot)
    }

    pub fn simulate(&mut self, now_ms: u64) {
        for entity in self.entities.values_mut() {
            entity.simulate(now_ms);
        }
    }

    pub fn spawn(
        &mut self,
        entity_id: EntityId,
        kind: EntityKind,
        position: [f32; 4],
        orientation: [f32; 4],
        scale: f32,
        material: u8,
        last_update_ms: u64,
    ) -> EntityId {
        let entity = EntityState {
            core: EntityCore {
                entity_id,
                class: EntityClass::Accent,
                position,
                orientation: normalize4_with_fallback(orientation, [0.0, 0.0, 1.0, 0.0]),
                velocity: [0.0, 0.0, 0.0, 0.0],
                last_update_ms,
            },
            kind,
            home_position: position,
            base_scale: scale,
            scale,
            material,
        };
        let prior = self.entities.insert(entity_id, entity);
        assert!(prior.is_none(), "duplicate entity id: {entity_id}");
        entity_id
    }
}
