use alloc::string::String;
use alloc::vec::Vec;
use polychora_plugin_api::content_ids::*;
use polychora_plugin_api::gui_abi::*;
use polychora_plugin_api::side_effects::{SideEffect, WasmCallResult};

const CHEST_SLOTS: u32 = 27;
const CHEST_COLUMNS: u32 = 9;

/// Internal state serialized into `gui_state` while the chest GUI is open.
/// Contains the chest inventory slots and player inventory slots together.
#[derive(serde::Serialize, serde::Deserialize)]
struct ChestGuiState {
    position: [i64; 4],
    chest_slots: Vec<ItemSlot>,
    player_slots: Vec<ItemSlot>,
}

// Well-known item type constants (must match host's item_types.rs).
const ITEM_SPAWN_EGG_NS: u32 = 0;
const ITEM_SPAWN_EGG_TYPE: u32 = 2;

/// CBOR-encoded spawn egg metadata (must match host's SpawnEggMeta).
#[derive(serde::Serialize, serde::Deserialize)]
struct SpawnEggMeta {
    entity_namespace: u32,
    entity_type: u32,
}

pub fn block_interact(input: &BlockInteractInput) -> WasmCallResult<BlockInteractOutput> {
    match input.block_type {
        BLOCK_CHEST => WasmCallResult::new(chest_interact(input)),
        BLOCK_SPAWNER => spawner_interact(input),
        BLOCK_BLUEPRINT_DISPENSER => blueprint_dispenser_interact(),
        _ => WasmCallResult::new(BlockInteractOutput::Nothing),
    }
}

fn spawner_interact(input: &BlockInteractInput) -> WasmCallResult<BlockInteractOutput> {
    // Check if the player is holding a spawn egg
    let held = input
        .player_inventory
        .get(input.held_item_index as usize);
    let held = match held {
        Some(slot) if !slot.is_empty() => slot,
        _ => return WasmCallResult::new(BlockInteractOutput::Nothing),
    };
    if held.item_ns != ITEM_SPAWN_EGG_NS || held.item_type != ITEM_SPAWN_EGG_TYPE {
        return WasmCallResult::new(BlockInteractOutput::Nothing);
    }

    // Decode spawn egg metadata (CBOR)
    let meta: SpawnEggMeta = match ciborium::from_reader(held.data.as_slice()) {
        Ok(m) => m,
        Err(_) => return WasmCallResult::new(BlockInteractOutput::Nothing),
    };

    // Build updated spawner metadata: reset spawn count, set entity type
    let new_metadata = {
        let mut out = Vec::with_capacity(20);
        out.extend_from_slice(&0u64.to_le_bytes()); // last_spawn_ms = 0
        out.extend_from_slice(&0u32.to_le_bytes()); // spawn_count = 0
        out.extend_from_slice(&meta.entity_namespace.to_le_bytes());
        out.extend_from_slice(&meta.entity_type.to_le_bytes());
        out
    };

    WasmCallResult::with_effects(
        BlockInteractOutput::Nothing,
        alloc::vec![
            SideEffect::UpdateBlockMetadata {
                metadata: new_metadata,
            },
            SideEffect::ConsumeHeldItem { count: 1 },
        ],
    )
}

fn blueprint_dispenser_interact() -> WasmCallResult<BlockInteractOutput> {
    // Build a 5×5×5×1 hollow frame (edges only) as a block list.
    const SIZE: i32 = 5;
    let mut blocks: Vec<BlueprintBlock> = Vec::new();

    for x in 0..SIZE {
        for y in 0..SIZE {
            for z in 0..SIZE {
                let edge_count = [x, y, z]
                    .iter()
                    .filter(|&&v| v == 0 || v == SIZE - 1)
                    .count();
                if edge_count >= 2 {
                    let (ns, bt) = if edge_count >= 3 {
                        (CONTENT_NS, BLOCK_LIGHT)
                    } else {
                        (CONTENT_NS, BLOCK_STONE)
                    };
                    blocks.push(BlueprintBlock {
                        offset: [x, y, z, 0],
                        namespace: ns,
                        block_type: bt,
                    });
                }
            }
        }
    }

    let blueprint_meta = BlueprintMeta { blocks };
    let mut item_data = Vec::new();
    ciborium::into_writer(&blueprint_meta, &mut item_data).unwrap();

    WasmCallResult::with_effects(
        BlockInteractOutput::Nothing,
        alloc::vec![SideEffect::GiveItem {
            item_ns: 0,
            item_type: 3,
            item_data,
            count: 1,
        }],
    )
}

/// A single block placement in a blueprint (must match host's BlueprintBlock).
#[derive(serde::Serialize, serde::Deserialize)]
struct BlueprintBlock {
    offset: [i32; 4],
    namespace: u32,
    block_type: u32,
}

/// Blueprint metadata (must match host's BlueprintMeta).
#[derive(serde::Serialize, serde::Deserialize)]
struct BlueprintMeta {
    blocks: Vec<BlueprintBlock>,
}

fn chest_interact(input: &BlockInteractInput) -> BlockInteractOutput {
    let chest_slots: Vec<ItemSlot> = if input.metadata.is_empty() {
        (0..CHEST_SLOTS).map(|_| ItemSlot::empty()).collect()
    } else {
        postcard::from_bytes(&input.metadata).unwrap_or_else(|_| {
            (0..CHEST_SLOTS).map(|_| ItemSlot::empty()).collect()
        })
    };

    let state = ChestGuiState {
        position: input.position,
        chest_slots: chest_slots.clone(),
        player_slots: input.player_inventory.clone(),
    };
    let gui_state = postcard::to_allocvec(&state).unwrap_or_default();

    // Build combined slot list: chest slots, then player slots
    let mut all_slots = chest_slots;
    all_slots.extend(input.player_inventory.iter().cloned());

    BlockInteractOutput::OpenGui(GuiDescription {
        title: String::from("Chest"),
        slot_groups: alloc::vec![SlotGroup {
            label: None,
            slot_start: 0,
            slot_end: CHEST_SLOTS,
            columns: CHEST_COLUMNS,
        }],
        slots: all_slots,
        gui_state,
        show_player_inventory: true,
        tick_interval_ms: None,
    })
}

pub fn gui_action(input: &GuiActionInput) -> WasmCallResult<GuiActionOutput> {
    let mut state: ChestGuiState = match postcard::from_bytes(&input.gui_state) {
        Ok(s) => s,
        Err(_) => {
            return WasmCallResult::new(GuiActionOutput {
                accepted: false,
                slots: Vec::new(),
                gui_state: input.gui_state.clone(),
            })
        }
    };

    let mut all_slots = state.chest_slots.clone();
    all_slots.extend(state.player_slots.iter().cloned());

    match &input.action {
        GuiAction::MoveStack {
            from_slot,
            to_slot,
            count,
        } => {
            let from = *from_slot as usize;
            let to = *to_slot as usize;
            let count = *count;

            if from >= all_slots.len() || to >= all_slots.len() || from == to {
                return WasmCallResult::new(GuiActionOutput {
                    accepted: false,
                    slots: all_slots,
                    gui_state: input.gui_state.clone(),
                });
            }

            let accepted = move_items(&mut all_slots, from, to, count);

            // Split back into chest and player
            let chest_end = CHEST_SLOTS as usize;
            state.chest_slots = all_slots[..chest_end].to_vec();
            state.player_slots = all_slots[chest_end..].to_vec();

            let gui_state = postcard::to_allocvec(&state).unwrap_or_default();
            WasmCallResult::new(GuiActionOutput {
                accepted,
                slots: all_slots,
                gui_state,
            })
        }
    }
}

pub fn gui_close(input: &GuiCloseInput) -> WasmCallResult<GuiCloseOutput> {
    let state: ChestGuiState = match postcard::from_bytes(&input.gui_state) {
        Ok(s) => s,
        Err(_) => {
            return WasmCallResult::new(GuiCloseOutput {
                metadata: Vec::new(),
                player_inventory: Vec::new(),
            })
        }
    };

    let metadata = postcard::to_allocvec(&state.chest_slots).unwrap_or_default();
    WasmCallResult::new(GuiCloseOutput {
        metadata,
        player_inventory: state.player_slots,
    })
}

/// Move `count` items from `from` to `to` within a flat slot array.
fn move_items(slots: &mut [ItemSlot], from: usize, to: usize, count: u32) -> bool {
    if slots[from].is_empty() || count == 0 {
        return false;
    }

    let actual_count = count.min(slots[from].count);

    if slots[to].is_empty() {
        // Move into empty slot
        slots[to] = slots[from].clone();
        slots[to].count = actual_count;
        slots[from].count -= actual_count;
        if slots[from].count == 0 {
            slots[from] = ItemSlot::empty();
        }
        true
    } else if slots[to].item_ns == slots[from].item_ns
        && slots[to].item_type == slots[from].item_type
        && slots[to].data == slots[from].data
    {
        // Stack merge (cap at 64 for now)
        let space = 64u32.saturating_sub(slots[to].count);
        let transfer = actual_count.min(space);
        if transfer == 0 {
            return false;
        }
        slots[to].count += transfer;
        slots[from].count -= transfer;
        if slots[from].count == 0 {
            slots[from] = ItemSlot::empty();
        }
        true
    } else if actual_count == slots[from].count {
        // Swap — moving entire stack into occupied slot of different type
        slots.swap(from, to);
        true
    } else {
        // Can't partial-move into a slot with a different item type
        false
    }
}
