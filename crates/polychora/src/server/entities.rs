use crate::content_registry::ContentRegistry;
use crate::shared::entity_types::SimulationMode;
use crate::shared::protocol::{Entity, EntitySnapshot};
use crate::shared::wasm::{WasmPluginManager, WasmPluginSlot};
use polychora_plugin_api::entity_tick_abi::{EntityTickInput, EntityTickOutput};
use polychora_plugin_api::opcodes::OP_ENTITY_TICK;
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
    fn simulate(
        &mut self,
        now_ms: u64,
        registry: &ContentRegistry,
        wasm_manager: &mut Option<WasmPluginManager>,
    ) -> bool {
        // Only tick entities with Parametric simulation mode
        let sim_config = registry.sim_config(self.entity.namespace, self.entity.entity_type);
        let is_parametric = sim_config
            .map(|c| c.mode == SimulationMode::Parametric)
            .unwrap_or(false);
        if !is_parametric {
            return false;
        }

        let previous_position = self.entity.pose.position;
        let previous_orientation = self.entity.pose.orientation;
        let previous_scale = self.entity.pose.scale;

        // Build EntityTickInput for the WASM plugin
        let tick_input = EntityTickInput {
            entity_ns: self.entity.namespace,
            entity_type: self.entity.entity_type,
            entity_id: self.entity_id,
            position: self.entity.pose.position,
            home_position: self.home_position,
            scale: self.base_scale,
            now_ms,
            ..Default::default()
        };

        let (next_position, next_orientation, next_scale) =
            if let Some(pose) = wasm_entity_set_pose(wasm_manager, &tick_input) {
                pose
            } else {
                // Fallback: no animation (unknown entity or no WASM)
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

/// Call OP_ENTITY_TICK via WASM and extract a SetPose result.
fn wasm_entity_set_pose(
    wasm_manager: &mut Option<WasmPluginManager>,
    input: &EntityTickInput,
) -> Option<([f32; 4], [f32; 4], f32)> {
    let manager = wasm_manager.as_mut()?;
    let input_bytes = postcard::to_allocvec(input).ok()?;
    let result = manager
        .call_slot(WasmPluginSlot::EntitySimulation, OP_ENTITY_TICK as i32, &input_bytes)
        .ok()??;
    let output: EntityTickOutput = postcard::from_bytes(&result.invocation.output).ok()?;
    match output {
        EntityTickOutput::SetPose { position, orientation, scale } => {
            // Validate finite values
            if position.iter().all(|v| v.is_finite())
                && orientation.iter().all(|v| v.is_finite())
                && scale.is_finite()
            {
                Some((position, orientation, scale))
            } else {
                None
            }
        }
        EntityTickOutput::Steer { .. } => None, // Parametric entities should not return Steer
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

    pub fn simulate(
        &mut self,
        now_ms: u64,
        registry: &ContentRegistry,
        wasm_manager: &mut Option<WasmPluginManager>,
    ) -> Vec<EntityId> {
        let mut moved = Vec::new();
        for entity in self.entities.values_mut() {
            if entity.simulate(now_ms, registry, wasm_manager) {
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
