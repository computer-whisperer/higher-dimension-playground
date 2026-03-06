use alloc::vec;
use alloc::vec::Vec;
use polychora_plugin_api::block_tick_abi::{BlockTickAction, BlockTickInput, BlockTickOutput};
use polychora_plugin_api::content_ids::*;

/// Spawner state serialized into block metadata.
struct SpawnerState {
    /// Server time (ms) of the last spawn.
    last_spawn_ms: u64,
    /// Number of entities spawned so far by this instance.
    spawn_count: u32,
}

impl SpawnerState {
    fn decode(bytes: &[u8]) -> Self {
        if bytes.len() >= 12 {
            let last_spawn_ms = u64::from_le_bytes(bytes[0..8].try_into().unwrap());
            let spawn_count = u32::from_le_bytes(bytes[8..12].try_into().unwrap());
            Self {
                last_spawn_ms,
                spawn_count,
            }
        } else {
            Self {
                last_spawn_ms: 0,
                spawn_count: 0,
            }
        }
    }

    fn encode(&self) -> Vec<u8> {
        let mut out = Vec::with_capacity(12);
        out.extend_from_slice(&self.last_spawn_ms.to_le_bytes());
        out.extend_from_slice(&self.spawn_count.to_le_bytes());
        out
    }
}

const SPAWNER_COOLDOWN_MS: u64 = 5000;
const SPAWNER_MAX_SPAWNS: u32 = 4;

pub fn block_tick(input: &BlockTickInput) -> BlockTickOutput {
    match input.block_type {
        BLOCK_SPAWNER => spawner_tick(input),
        _ => BlockTickOutput::default(),
    }
}

fn spawner_tick(input: &BlockTickInput) -> BlockTickOutput {
    let mut state = SpawnerState::decode(&input.metadata);

    if state.spawn_count >= SPAWNER_MAX_SPAWNS {
        return BlockTickOutput {
            metadata: state.encode(),
            actions: Vec::new(),
        };
    }

    if input.now_ms < state.last_spawn_ms + SPAWNER_COOLDOWN_MS {
        return BlockTickOutput {
            metadata: state.encode(),
            actions: Vec::new(),
        };
    }

    // Spawn a seeker above the spawner block
    state.last_spawn_ms = input.now_ms;
    state.spawn_count += 1;

    BlockTickOutput {
        metadata: state.encode(),
        actions: vec![BlockTickAction::SpawnEntity {
            entity_type: ENTITY_SEEKER,
            offset: [0.0, 1.5, 0.0, 0.0],
        }],
    }
}
