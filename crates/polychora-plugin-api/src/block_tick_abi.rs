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
///
/// Side effects (e.g. entity spawning) are returned via `WasmCallResult`
/// rather than in this struct directly.
#[derive(Serialize, Deserialize, Default)]
pub struct BlockTickOutput {
    /// Updated per-instance metadata to write back to the block.
    #[serde(default)]
    pub metadata: Vec<u8>,
}
