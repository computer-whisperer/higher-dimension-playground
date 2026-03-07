use polychora_plugin_api::gui_abi::{
    BlockInteractInput, BlockInteractOutput, GuiAction, GuiActionInput, GuiActionOutput,
    GuiCloseInput, GuiCloseOutput, ItemSlot,
};
use polychora_plugin_api::opcodes::{OP_BLOCK_INTERACT, OP_GUI_ACTION, OP_GUI_CLOSE};
use polychora_plugin_api::side_effects::{SideEffect, WasmCallResult};

use crate::shared::inventory::{Inventory, INVENTORY_SIZE};
use crate::shared::protocol::ItemStack;
use crate::shared::voxel::BlockData;
use crate::shared::wasm::{WasmPluginManager, WasmPluginSlot};

/// Active GUI session state for an open block interaction GUI.
pub struct BlockGuiSession {
    pub title: String,
    pub slot_groups: Vec<polychora_plugin_api::gui_abi::SlotGroup>,
    pub slots: Vec<ItemSlot>,
    pub gui_state: Vec<u8>,
    pub show_player_inventory: bool,
    pub tick_interval_ms: Option<u64>,
    /// Number of block-side slots (before player inventory slots).
    pub block_slot_count: u32,
    /// Position of the block in the world.
    pub block_position: [i64; 4],
    /// Block type info for metadata writeback.
    pub block_ns: u32,
    pub block_type: u32,
    /// Slot index the player has "picked up" (click-to-select, click-to-place).
    pub held_slot: Option<u32>,
}

/// Result of a block interaction that didn't open a GUI.
pub struct InteractSideEffects {
    pub side_effects: Vec<SideEffect>,
}

/// Convert player inventory to plugin ItemSlot format.
pub fn inventory_to_item_slots(inventory: &Inventory) -> Vec<ItemSlot> {
    (0..INVENTORY_SIZE)
        .map(|i| match inventory.slot(i) {
            Some(stack) => ItemSlot {
                item_ns: stack.item.namespace,
                item_type: stack.item.item_type,
                count: stack.count,
                data: stack.item.data.clone(),
            },
            None => ItemSlot::empty(),
        })
        .collect()
}

/// Convert plugin ItemSlots back to player inventory.
pub fn item_slots_to_inventory(slots: &[ItemSlot]) -> Inventory {
    let mut inv = Inventory::default();
    for (i, slot) in slots.iter().enumerate().take(INVENTORY_SIZE) {
        if slot.is_empty() {
            inv.set_slot(i, None);
        } else {
            inv.set_slot(
                i,
                Some(ItemStack {
                    item: crate::shared::protocol::Item {
                        namespace: slot.item_ns,
                        item_type: slot.item_type,
                        data: slot.data.clone(),
                    },
                    count: slot.count,
                }),
            );
        }
    }
    inv
}

/// Result of calling OP_BLOCK_INTERACT.
pub enum BlockInteractResult {
    /// No interaction occurred.
    Nothing,
    /// A GUI was opened.
    OpenGui(BlockGuiSession),
    /// Interaction handled without GUI; side effects need processing.
    Handled(InteractSideEffects),
}

/// Try to interact with a block by calling OP_BLOCK_INTERACT.
pub fn try_block_interact(
    wasm: &mut WasmPluginManager,
    block: &BlockData,
    position: [i64; 4],
    inventory: &Inventory,
    held_item_index: u32,
) -> BlockInteractResult {
    let input = BlockInteractInput {
        block_ns: block.namespace,
        block_type: block.block_type,
        position,
        metadata: block.extra_data.clone(),
        player_inventory: inventory_to_item_slots(inventory),
        held_item_index,
    };

    let input_bytes = match postcard::to_allocvec(&input) {
        Ok(b) => b,
        Err(_) => return BlockInteractResult::Nothing,
    };
    let result = match wasm.call_slot(
        WasmPluginSlot::ModelLogic,
        OP_BLOCK_INTERACT as i32,
        &input_bytes,
    ) {
        Ok(Some(r)) => r,
        Ok(None) => {
            eprintln!("OP_BLOCK_INTERACT: WASM call returned None (no plugin loaded?)");
            return BlockInteractResult::Nothing;
        }
        Err(e) => {
            eprintln!("OP_BLOCK_INTERACT: WASM call failed: {e}");
            return BlockInteractResult::Nothing;
        }
    };

    let wrapped: WasmCallResult<BlockInteractOutput> =
        match postcard::from_bytes(&result.invocation.output) {
            Ok(o) => o,
            Err(e) => {
                eprintln!(
                    "OP_BLOCK_INTERACT: failed to deserialize output ({} bytes): {e}",
                    result.invocation.output.len(),
                );
                return BlockInteractResult::Nothing;
            }
        };

    match wrapped.response {
        BlockInteractOutput::Nothing => {
            if wrapped.side_effects.is_empty() {
                BlockInteractResult::Nothing
            } else {
                BlockInteractResult::Handled(InteractSideEffects {
                    side_effects: wrapped.side_effects,
                })
            }
        }
        BlockInteractOutput::OpenGui(desc) => {
            let block_slot_count = desc
                .slot_groups
                .iter()
                .map(|g| g.slot_end)
                .max()
                .unwrap_or(0);
            BlockInteractResult::OpenGui(BlockGuiSession {
                title: desc.title,
                slot_groups: desc.slot_groups,
                slots: desc.slots,
                gui_state: desc.gui_state,
                show_player_inventory: desc.show_player_inventory,
                tick_interval_ms: desc.tick_interval_ms,
                block_slot_count,
                block_position: position,
                block_ns: block.namespace,
                block_type: block.block_type,
                held_slot: None,
            })
        }
    }
}

/// Send a GUI action to the plugin and update session state.
/// Returns true if the action was accepted.
pub fn send_gui_action(
    wasm: &mut WasmPluginManager,
    session: &mut BlockGuiSession,
    action: GuiAction,
) -> bool {
    let input = GuiActionInput {
        gui_state: session.gui_state.clone(),
        action,
    };

    let input_bytes = match postcard::to_allocvec(&input) {
        Ok(b) => b,
        Err(_) => return false,
    };

    let result = match wasm.call_slot(
        WasmPluginSlot::ModelLogic,
        OP_GUI_ACTION as i32,
        &input_bytes,
    ) {
        Ok(Some(r)) => r,
        _ => return false,
    };

    let wrapped: WasmCallResult<GuiActionOutput> =
        match postcard::from_bytes(&result.invocation.output) {
            Ok(o) => o,
            Err(_) => return false,
        };

    session.slots = wrapped.response.slots;
    session.gui_state = wrapped.response.gui_state;
    wrapped.response.accepted
}

/// Close the GUI and get back updated metadata + player inventory.
pub fn close_gui(
    wasm: &mut WasmPluginManager,
    session: &BlockGuiSession,
) -> Option<GuiCloseOutput> {
    let input = GuiCloseInput {
        gui_state: session.gui_state.clone(),
    };

    let input_bytes = postcard::to_allocvec(&input).ok()?;
    let result = wasm
        .call_slot(
            WasmPluginSlot::ModelLogic,
            OP_GUI_CLOSE as i32,
            &input_bytes,
        )
        .ok()??;

    let wrapped: WasmCallResult<GuiCloseOutput> =
        postcard::from_bytes(&result.invocation.output).ok()?;
    Some(wrapped.response)
}
