use alloc::vec::Vec;
use serde::{Deserialize, Serialize};

/// A side effect returned by a WASM plugin call that the host should execute.
///
/// Each opcode has a whitelist of allowed side effects; the host ignores any
/// that are not permitted for the current opcode.
#[derive(Clone, Debug, Serialize, Deserialize)]
pub enum SideEffect {
    /// Spawn an entity near the block.
    SpawnEntity {
        entity_type_ns: u32,
        entity_type: u32,
        /// Offset from the block position (in voxels).
        offset: [f32; 4],
    },
    /// Update the block's metadata (extra_data) at its position.
    UpdateBlockMetadata { metadata: Vec<u8> },
    /// Consume items from the player's held hotbar slot.
    ConsumeHeldItem { count: u32 },
    /// Give an item to the player, adding it to their inventory.
    GiveItem {
        item_ns: u32,
        item_type: u32,
        item_data: Vec<u8>,
        count: u32,
    },
}

/// Generic wrapper for all WASM opcode outputs.
///
/// Every opcode returns `WasmCallResult<T>` where `T` is the opcode-specific
/// response type. Side effects are processed by the host after the call.
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct WasmCallResult<T> {
    pub response: T,
    #[serde(default)]
    pub side_effects: Vec<SideEffect>,
}

impl<T: Default> Default for WasmCallResult<T> {
    fn default() -> Self {
        Self {
            response: T::default(),
            side_effects: Vec::new(),
        }
    }
}

impl<T> WasmCallResult<T> {
    pub fn new(response: T) -> Self {
        Self {
            response,
            side_effects: Vec::new(),
        }
    }

    pub fn with_effects(response: T, side_effects: Vec<SideEffect>) -> Self {
        Self {
            response,
            side_effects,
        }
    }
}
