use crate::shared::protocol::ItemStack;
use serde::{Deserialize, Serialize};

pub const INVENTORY_COLS: usize = 9;
pub const INVENTORY_ROWS: usize = 4;
pub const INVENTORY_SIZE: usize = INVENTORY_ROWS * INVENTORY_COLS; // 36
pub const HOTBAR_SIZE: usize = INVENTORY_COLS; // 9
pub const MAX_STACK_SIZE: u32 = 64;

#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub enum GameMode {
    Creative,
    Survival,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum InventoryTab {
    Creative,
    Survival,
}

/// 36-slot inventory: row 0 (slots 0..9) = hotbar, rows 1-3 (slots 9..36) = main.
#[derive(Clone, Debug)]
pub struct Inventory {
    slots: [Option<ItemStack>; INVENTORY_SIZE],
}

impl Default for Inventory {
    fn default() -> Self {
        Self {
            slots: std::array::from_fn(|_| None),
        }
    }
}

impl Inventory {
    /// Creative default: hotbar pre-populated with the standard 9 blocks, rest empty.
    pub fn default_creative() -> Self {
        use polychora_plugin_api::content_ids::*;
        let mut inv = Self::default();
        let defaults = [
            (CONTENT_NS, BLOCK_YELLOW_GREEN),
            (CONTENT_NS, BLOCK_STONE),
            (CONTENT_NS, BLOCK_COBBLESTONE),
            (CONTENT_NS, BLOCK_DIRT),
            (CONTENT_NS, BLOCK_OAK_PLANKS),
            (CONTENT_NS, BLOCK_WHITE),
            (CONTENT_NS, BLOCK_LIGHT),
            (CONTENT_NS, BLOCK_RED),
            (CONTENT_NS, BLOCK_GREEN),
        ];
        for (i, (ns, bt)) in defaults.into_iter().enumerate() {
            inv.slots[i] = Some(ItemStack::block(ns, bt, 1, 0));
        }
        inv
    }

    pub fn slot(&self, i: usize) -> Option<&ItemStack> {
        self.slots.get(i).and_then(|s| s.as_ref())
    }

    pub fn hotbar_slot(&self, i: usize) -> &Option<ItemStack> {
        &self.slots[i.min(HOTBAR_SIZE - 1)]
    }

    pub fn set_slot(&mut self, i: usize, stack: Option<ItemStack>) {
        if i < INVENTORY_SIZE {
            self.slots[i] = stack;
        }
    }

    /// Try to add a stack to the inventory. Merges into matching stacks first,
    /// then fills empty slots. Returns remainder that didn't fit (None if all added).
    pub fn try_add(&mut self, mut stack: ItemStack) -> Option<ItemStack> {
        // First pass: merge into existing matching stacks
        for slot in self.slots.iter_mut() {
            if stack.count == 0 {
                return None;
            }
            if let Some(existing) = slot {
                if existing.item == stack.item && existing.count < MAX_STACK_SIZE {
                    let space = MAX_STACK_SIZE - existing.count;
                    let transfer = stack.count.min(space);
                    existing.count += transfer;
                    stack.count -= transfer;
                }
            }
        }
        if stack.count == 0 {
            return None;
        }

        // Second pass: fill empty slots
        for slot in self.slots.iter_mut() {
            if stack.count == 0 {
                return None;
            }
            if slot.is_none() {
                let transfer = stack.count.min(MAX_STACK_SIZE);
                *slot = Some(ItemStack {
                    item: stack.item.clone(),
                    count: transfer,
                });
                stack.count -= transfer;
            }
        }

        if stack.count == 0 {
            None
        } else {
            Some(stack)
        }
    }

    /// Try to merge a stack into a specific slot. Returns remainder.
    pub fn try_add_at(&mut self, index: usize, stack: ItemStack) -> Option<ItemStack> {
        if index >= INVENTORY_SIZE {
            return Some(stack);
        }
        match &mut self.slots[index] {
            Some(existing) if existing.item == stack.item => {
                let space = MAX_STACK_SIZE - existing.count;
                let transfer = stack.count.min(space);
                existing.count += transfer;
                let remaining = stack.count - transfer;
                if remaining > 0 {
                    Some(ItemStack {
                        item: stack.item,
                        count: remaining,
                    })
                } else {
                    None
                }
            }
            None => {
                self.slots[index] = Some(stack);
                None
            }
            _ => Some(stack),
        }
    }

    /// Subtract 1 from a slot's count. Clears the slot at 0. Returns true if decremented.
    pub fn decrement_slot(&mut self, i: usize) -> bool {
        if i >= INVENTORY_SIZE {
            return false;
        }
        if let Some(stack) = &mut self.slots[i] {
            if stack.count > 1 {
                stack.count -= 1;
            } else {
                self.slots[i] = None;
            }
            true
        } else {
            false
        }
    }

    /// Update the scale_exp of the block item in a slot. No-op if the slot is
    /// empty or not a block item.
    pub fn update_slot_scale(&mut self, index: usize, scale_exp: i8) {
        if index >= INVENTORY_SIZE {
            return;
        }
        if let Some(stack) = &mut self.slots[index] {
            use crate::shared::item_types::{BlockItemMeta, ITEM_BLOCK};
            if (stack.item.namespace, stack.item.item_type) != ITEM_BLOCK {
                return;
            }
            if let Some(mut meta) = BlockItemMeta::decode(&stack.item.data) {
                meta.scale_exp = scale_exp;
                stack.item.data = meta.encode();
            }
        }
    }

    pub fn swap_slots(&mut self, a: usize, b: usize) {
        if a < INVENTORY_SIZE && b < INVENTORY_SIZE {
            self.slots.swap(a, b);
        }
    }

    /// Find first slot containing a block with the given (namespace, block_type).
    pub fn find_block(&self, ns: u32, bt: u32) -> Option<usize> {
        self.slots.iter().position(|slot| {
            slot.as_ref()
                .and_then(|s| s.block_type_key())
                .is_some_and(|(n, b)| n == ns && b == bt)
        })
    }

    /// Serialize to bytes for PlayerRecord.inventory_payload.
    pub fn to_payload(&self) -> Vec<u8> {
        // Serialize as Vec to avoid serde's fixed-array size limitation.
        let vec: Vec<Option<ItemStack>> = self.slots.to_vec();
        postcard::to_stdvec(&vec).unwrap_or_default()
    }

    /// Deserialize from PlayerRecord.inventory_payload.
    pub fn from_payload(data: &[u8]) -> Option<Self> {
        let vec: Vec<Option<ItemStack>> = postcard::from_bytes(data).ok()?;
        if vec.len() != INVENTORY_SIZE {
            return None;
        }
        let mut slots: [Option<ItemStack>; INVENTORY_SIZE] = std::array::from_fn(|_| None);
        for (i, item) in vec.into_iter().enumerate() {
            slots[i] = item;
        }
        Some(Self { slots })
    }

    /// Convert from legacy u8 material tokens (settings.json compat).
    pub fn from_legacy_hotbar_tokens(tokens: &[u8; 9]) -> Self {
        let mut inv = Self::default();
        for (i, &token) in tokens.iter().enumerate() {
            let block = crate::content_registry::block_data_from_material_token(token);
            if !block.is_air() {
                inv.slots[i] = Some(ItemStack::block(block.namespace, block.block_type, 1, 0));
            }
        }
        inv
    }

    /// Convert hotbar row back to legacy u8 material tokens.
    pub fn to_legacy_hotbar_tokens(&self) -> [u8; 9] {
        let mut tokens = [0u8; 9];
        for (i, slot) in self.slots[..HOTBAR_SIZE].iter().enumerate() {
            let block = slot
                .as_ref()
                .and_then(|s| s.to_block_data())
                .unwrap_or(crate::shared::voxel::BlockData::AIR);
            tokens[i] = crate::content_registry::material_token_from_block_data(&block);
        }
        tokens
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use polychora_plugin_api::content_ids::*;

    #[test]
    fn try_add_stacking() {
        let mut inv = Inventory::default();
        inv.set_slot(0, Some(ItemStack::block(CONTENT_NS, BLOCK_STONE, 60, 0)));

        // Should merge into slot 0
        let remainder = inv.try_add(ItemStack::block(CONTENT_NS, BLOCK_STONE, 3, 0));
        assert!(remainder.is_none());
        assert_eq!(inv.slot(0).unwrap().count, 63);
    }

    #[test]
    fn try_add_overflow() {
        let mut inv = Inventory::default();
        inv.set_slot(0, Some(ItemStack::block(CONTENT_NS, BLOCK_STONE, 60, 0)));

        // 10 merges 4 into slot 0, 6 into empty slot 1
        let remainder = inv.try_add(ItemStack::block(CONTENT_NS, BLOCK_STONE, 10, 0));
        assert!(remainder.is_none());
        assert_eq!(inv.slot(0).unwrap().count, 64);
        assert_eq!(inv.slot(1).unwrap().count, 6);
    }

    #[test]
    fn try_add_full_inventory() {
        let mut inv = Inventory::default();
        for i in 0..INVENTORY_SIZE {
            inv.set_slot(
                i,
                Some(ItemStack::block(CONTENT_NS, BLOCK_STONE, MAX_STACK_SIZE, 0)),
            );
        }
        let remainder = inv.try_add(ItemStack::block(CONTENT_NS, BLOCK_DIRT, 1, 0));
        assert!(remainder.is_some());
        assert_eq!(remainder.unwrap().count, 1);
    }

    #[test]
    fn decrement_slot_works() {
        let mut inv = Inventory::default();
        inv.set_slot(0, Some(ItemStack::block(CONTENT_NS, BLOCK_STONE, 3, 0)));
        assert!(inv.decrement_slot(0));
        assert_eq!(inv.slot(0).unwrap().count, 2);
        assert!(inv.decrement_slot(0));
        assert_eq!(inv.slot(0).unwrap().count, 1);
        assert!(inv.decrement_slot(0));
        assert!(inv.slot(0).is_none());
    }

    #[test]
    fn payload_round_trip() {
        let mut inv = Inventory::default_creative();
        inv.set_slot(9, Some(ItemStack::block(CONTENT_NS, BLOCK_STONE, 42, 0)));

        let payload = inv.to_payload();
        let restored = Inventory::from_payload(&payload).expect("should deserialize");
        for i in 0..INVENTORY_SIZE {
            assert_eq!(
                inv.slot(i).map(|s| (s.block_type_key(), s.count)),
                restored.slot(i).map(|s| (s.block_type_key(), s.count)),
                "slot {i} mismatch"
            );
        }
    }

    #[test]
    fn legacy_token_round_trip() {
        let inv = Inventory::default_creative();
        let tokens = inv.to_legacy_hotbar_tokens();
        let restored = Inventory::from_legacy_hotbar_tokens(&tokens);
        for i in 0..HOTBAR_SIZE {
            assert_eq!(
                inv.slot(i).and_then(|s| s.block_type_key()),
                restored.slot(i).and_then(|s| s.block_type_key()),
                "hotbar slot {i} mismatch"
            );
        }
    }

    #[test]
    fn find_block_works() {
        let inv = Inventory::default_creative();
        let idx = inv.find_block(CONTENT_NS, BLOCK_STONE);
        assert_eq!(idx, Some(1)); // Stone is the second default
    }

    #[test]
    fn swap_slots_works() {
        let mut inv = Inventory::default();
        inv.set_slot(0, Some(ItemStack::block(CONTENT_NS, BLOCK_STONE, 1, 0)));
        inv.set_slot(1, Some(ItemStack::block(CONTENT_NS, BLOCK_DIRT, 5, 0)));
        inv.swap_slots(0, 1);
        assert_eq!(
            inv.slot(0).unwrap().block_type_key(),
            Some((CONTENT_NS, BLOCK_DIRT))
        );
        assert_eq!(
            inv.slot(1).unwrap().block_type_key(),
            Some((CONTENT_NS, BLOCK_STONE))
        );
    }
}
