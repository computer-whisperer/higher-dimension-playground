use alloc::string::String;
use alloc::vec::Vec;
use serde::{Deserialize, Serialize};

// ---------------------------------------------------------------------------
// OP_BLOCK_INTERACT
// ---------------------------------------------------------------------------

/// Input to `OP_BLOCK_INTERACT` — sent when a player interacts with a block.
#[derive(Serialize, Deserialize, Default)]
pub struct BlockInteractInput {
    pub block_ns: u32,
    pub block_type: u32,
    pub position: [i64; 4],
    pub metadata: Vec<u8>,
    /// The player's inventory slots, serialized by the host.
    pub player_inventory: Vec<ItemSlot>,
    /// Index of the player's currently selected hotbar slot.
    #[serde(default)]
    pub held_item_index: u32,
}

/// Output from `OP_BLOCK_INTERACT`.
#[derive(Serialize, Deserialize)]
pub enum BlockInteractOutput {
    /// No GUI to open — interaction had no effect.
    Nothing,
    /// Open a GUI for this block.
    OpenGui(GuiDescription),
}

/// Describes a GUI that the host should render for the player.
#[derive(Serialize, Deserialize)]
pub struct GuiDescription {
    pub title: String,
    pub slot_groups: Vec<SlotGroup>,
    pub slots: Vec<ItemSlot>,
    /// Opaque plugin state, round-tripped through subsequent GUI calls.
    pub gui_state: Vec<u8>,
    /// Whether the host should display the player's inventory below the
    /// block's GUI slots.
    pub show_player_inventory: bool,
    /// If set, the host calls `OP_GUI_TICK` at this interval while the GUI
    /// is open. `None` means no ticking.
    pub tick_interval_ms: Option<u64>,
}

/// A named group of slots within the GUI layout.
#[derive(Serialize, Deserialize)]
pub struct SlotGroup {
    pub label: Option<String>,
    /// Inclusive start index into `GuiDescription::slots`.
    pub slot_start: u32,
    /// Exclusive end index into `GuiDescription::slots`.
    pub slot_end: u32,
    /// Number of columns for grid layout.
    pub columns: u32,
}

/// A single item slot in the GUI. All-zero means empty.
#[derive(Clone, Default, Serialize, Deserialize)]
pub struct ItemSlot {
    pub item_ns: u32,
    pub item_type: u32,
    pub count: u32,
    /// Opaque item data (e.g. BlockItemMeta encoded as CBOR).
    #[serde(default)]
    pub data: Vec<u8>,
}

impl ItemSlot {
    pub fn is_empty(&self) -> bool {
        self.count == 0
    }

    pub fn empty() -> Self {
        Self::default()
    }
}

// ---------------------------------------------------------------------------
// OP_GUI_TICK
// ---------------------------------------------------------------------------

/// Input to `OP_GUI_TICK` — periodic update while a GUI is open.
#[derive(Serialize, Deserialize)]
pub struct GuiTickInput {
    pub gui_state: Vec<u8>,
    pub now_ms: u64,
}

/// Output from `OP_GUI_TICK`.
#[derive(Serialize, Deserialize)]
pub struct GuiTickOutput {
    pub slots: Vec<ItemSlot>,
    pub gui_state: Vec<u8>,
}

// ---------------------------------------------------------------------------
// OP_GUI_ACTION
// ---------------------------------------------------------------------------

/// Input to `OP_GUI_ACTION` — player performed an action in the GUI.
#[derive(Serialize, Deserialize)]
pub struct GuiActionInput {
    pub gui_state: Vec<u8>,
    pub action: GuiAction,
}

/// A player-initiated action within an open GUI.
#[derive(Serialize, Deserialize)]
pub enum GuiAction {
    /// Move `count` items from one slot to another.
    /// Slot indices span both block GUI slots and (if shown) player inventory.
    MoveStack {
        from_slot: u32,
        to_slot: u32,
        count: u32,
    },
}

/// Output from `OP_GUI_ACTION`.
#[derive(Serialize, Deserialize)]
pub struct GuiActionOutput {
    /// Whether the action was accepted. If false, the host should revert
    /// the visual change.
    pub accepted: bool,
    pub slots: Vec<ItemSlot>,
    pub gui_state: Vec<u8>,
}

// ---------------------------------------------------------------------------
// OP_GUI_CLOSE
// ---------------------------------------------------------------------------

/// Input to `OP_GUI_CLOSE` — player closed the GUI.
#[derive(Serialize, Deserialize)]
pub struct GuiCloseInput {
    pub gui_state: Vec<u8>,
}

/// Output from `OP_GUI_CLOSE`.
#[derive(Serialize, Deserialize)]
pub struct GuiCloseOutput {
    /// Updated block metadata to persist.
    pub metadata: Vec<u8>,
    /// Updated player inventory to write back.
    pub player_inventory: Vec<ItemSlot>,
}
