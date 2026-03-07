use super::*;
use crate::shared::spatial::ChunkCoord;
use crate::shared::voxel::{self, BlockData};
use crate::shared::wasm::{WasmPluginManager, WasmPluginSlot};
use polychora_plugin_api::block_tick_abi::{BlockTickInput, BlockTickOutput};
use polychora_plugin_api::opcodes::OP_BLOCK_TICK;
use polychora_plugin_api::side_effects::{SideEffect, WasmCallResult};

/// An action produced by a block tick that the server loop should execute.
pub(super) enum BlockTickSpawnAction {
    SpawnEntity {
        namespace: u32,
        entity_type: u32,
        position: [f32; 4],
    },
}

/// Read the block at a scale-0 voxel position from the world.
fn read_block_at(state: &ServerState, position: [ChunkCoord; 4]) -> Option<BlockData> {
    let (chunk_key, voxel_index) = voxel::world_to_chunk_at_scale(
        position[0], position[1], position[2], position[3], 0,
    );
    if let Some((payload, _scale)) = state.world_chunk_at(chunk_key) {
        return Some(payload.block_at(voxel_index));
    }
    if let Some((payload, _scale)) = state.cached_chunk_at(chunk_key) {
        return Some(payload.block_at(voxel_index));
    }
    state
        .world_effective_chunk(chunk_key, false)
        .map(|p| p.block_at(voxel_index))
}

/// Run one block tick pass using the world cache's ticking block index.
///
/// Iterates all ticking blocks tracked by `ServerWorldCache`, checks intervals
/// and activation radius, calls WASM `OP_BLOCK_TICK`, writes metadata back,
/// and returns spawn actions for the caller to execute.
pub(super) fn run_block_ticks(
    state: &mut ServerState,
    wasm_manager: &mut Option<WasmPluginManager>,
    now_ms: u64,
) -> Vec<BlockTickSpawnAction> {
    let wasm = match wasm_manager.as_mut() {
        Some(w) => w,
        None => return Vec::new(),
    };

    let player_positions: Vec<[f32; 4]> = state
        .players
        .values()
        .filter_map(|player| state.entity_store.snapshot(player.entity_id))
        .map(|snapshot| snapshot.entity.pose.position)
        .collect();

    if player_positions.is_empty() {
        return Vec::new();
    }

    // Snapshot ticking blocks that are due. We collect first because we need
    // &mut state later to write metadata back.
    let content_registry = state.content_registry.clone();
    let to_tick: Vec<([ChunkCoord; 4], u32, u32)> = state
        .world_cache
        .ticking_blocks()
        .iter()
        .filter(|(_, entry)| {
            content_registry
                .block_tick_config(entry.namespace, entry.block_type)
                .map(|cfg| now_ms >= entry.last_tick_ms.saturating_add(cfg.interval_ms))
                .unwrap_or(false)
        })
        .map(|(pos, entry)| (*pos, entry.namespace, entry.block_type))
        .collect();

    let mut actions = Vec::new();

    for (pos, namespace, block_type) in to_tick {
        let Some(config) = content_registry.block_tick_config(namespace, block_type) else {
            continue;
        };

        let block_center = [
            pos[0].to_num::<f32>() + 0.5,
            pos[1].to_num::<f32>() + 0.5,
            pos[2].to_num::<f32>() + 0.5,
            pos[3].to_num::<f32>() + 0.5,
        ];

        // Activation radius check.
        let activation_radius_sq = config.activation_radius * config.activation_radius;
        let mut nearest_dist_sq = f32::MAX;
        let mut nearest_dir = [0.0f32; 4];
        for pp in &player_positions {
            let d = [
                pp[0] - block_center[0],
                pp[1] - block_center[1],
                pp[2] - block_center[2],
                pp[3] - block_center[3],
            ];
            let dist_sq = d[0] * d[0] + d[1] * d[1] + d[2] * d[2] + d[3] * d[3];
            if dist_sq < nearest_dist_sq {
                nearest_dist_sq = dist_sq;
                let dist = dist_sq.sqrt();
                if dist > 1e-6 {
                    nearest_dir = [d[0] / dist, d[1] / dist, d[2] / dist, d[3] / dist];
                }
            }
        }
        if nearest_dist_sq > activation_radius_sq {
            continue;
        }

        // Read current block from world to get extra_data.
        let Some(current_block) = read_block_at(state, pos) else {
            continue;
        };
        if current_block.namespace != namespace || current_block.block_type != block_type {
            continue;
        }

        let input = BlockTickInput {
            block_ns: namespace,
            block_type,
            position: pos.map(|c| c.to_num::<i64>()),
            metadata: current_block.extra_data.clone(),
            now_ms,
            nearest_player_distance: nearest_dist_sq.sqrt(),
            nearest_player_direction: nearest_dir,
        };

        let input_bytes = match postcard::to_allocvec(&input) {
            Ok(b) => b,
            Err(_) => continue,
        };
        let result = match wasm.call_slot(
            WasmPluginSlot::EntitySimulation,
            OP_BLOCK_TICK as i32,
            &input_bytes,
        ) {
            Ok(Some(r)) => r,
            _ => continue,
        };
        let wrapped: WasmCallResult<BlockTickOutput> =
            match postcard::from_bytes(&result.invocation.output) {
                Ok(o) => o,
                Err(_) => continue,
            };

        // Update last_tick_ms in the world cache.
        state.world_cache.update_tick_time(&pos, now_ms);

        // Write updated metadata back if changed.
        if wrapped.response.metadata != current_block.extra_data {
            let mut updated_block = current_block.clone();
            updated_block.extra_data = wrapped.response.metadata;
            let _ = state.apply_world_voxel_edit(pos, updated_block);
        }

        // Process side effects (whitelist: SpawnEntity only).
        for effect in wrapped.side_effects {
            match effect {
                SideEffect::SpawnEntity {
                    entity_type_ns,
                    entity_type,
                    offset,
                } => {
                    actions.push(BlockTickSpawnAction::SpawnEntity {
                        namespace: entity_type_ns,
                        entity_type,
                        position: [
                            block_center[0] + offset[0],
                            block_center[1] + offset[1],
                            block_center[2] + offset[2],
                            block_center[3] + offset[3],
                        ],
                    });
                }
                _ => {
                    eprintln!(
                        "Warning: disallowed side effect from OP_BLOCK_TICK at ({}, {}, {}, {})",
                        pos[0].to_num::<i32>(),
                        pos[1].to_num::<i32>(),
                        pos[2].to_num::<i32>(),
                        pos[3].to_num::<i32>(),
                    );
                }
            }
        }
    }

    actions
}
