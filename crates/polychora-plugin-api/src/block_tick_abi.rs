use alloc::vec::Vec;
use serde::{Deserialize, Serialize};

/// Input to `OP_BLOCK_TICK` — host sends this per ticking block instance.
#[derive(Serialize, Deserialize, Default)]
pub struct BlockTickInput {
    pub block_ns: u32,
    pub block_type: u32,
    /// World position of this block instance (chunk coordinates).
    #[serde(default)]
    pub position: [i64; 4],
    /// Opaque per-instance metadata, round-tripped from previous tick.
    /// Empty on the first tick after placement or world load.
    #[serde(default)]
    pub metadata: Vec<u8>,
    /// Server time in milliseconds.
    #[serde(default)]
    pub now_ms: u64,
    /// Distance to the nearest player, in voxels.
    #[serde(default)]
    pub nearest_player_distance: f32,
    /// Direction toward the nearest player (unit vector, 4D).
    #[serde(default)]
    pub nearest_player_direction: [f32; 4],
}

/// Output from `OP_BLOCK_TICK`.
#[derive(Serialize, Deserialize, Default)]
pub struct BlockTickOutput {
    /// Updated per-instance metadata to write back to the block.
    #[serde(default)]
    pub metadata: Vec<u8>,
    /// Actions the host should execute on behalf of this block.
    #[serde(default)]
    pub actions: Vec<BlockTickAction>,
}

/// An action returned by a block tick that the host should execute.
#[derive(Serialize, Deserialize)]
pub enum BlockTickAction {
    /// Spawn an entity near this block.
    SpawnEntity {
        /// Entity type ID (same namespace as the block).
        entity_type: u32,
        /// Offset from the block position (in voxels).
        #[serde(default)]
        offset: [f32; 4],
    },
}
