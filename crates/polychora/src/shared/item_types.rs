use crate::shared::protocol::{Item, ItemStack};
use crate::shared::voxel::{BlockData, TesseractOrientation};
use serde::{Deserialize, Serialize};

// ---------------------------------------------------------------------------
// Well-known item type constants: (namespace, item_type)
// ---------------------------------------------------------------------------

pub const ITEM_BLOCK_STACK: (u32, u32) = (0, 1);

// ---------------------------------------------------------------------------
// Item type registry entry
// ---------------------------------------------------------------------------

pub struct ItemTypeEntry {
    pub namespace: u32,
    pub item_type: u32,
    pub name: &'static str,
}

pub const ITEM_TYPE_ENTRIES: &[ItemTypeEntry] = &[ItemTypeEntry {
    namespace: 0,
    item_type: 1,
    name: "Block",
}];

// ---------------------------------------------------------------------------
// BlockStackMeta — CBOR schema for block_stack item data
// ---------------------------------------------------------------------------

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct BlockStackMeta {
    pub namespace: u32,
    pub block_type: u32,
    #[serde(default)]
    pub extra_data: Vec<u8>,
}

impl BlockStackMeta {
    pub fn encode(&self) -> Vec<u8> {
        let mut buf = Vec::new();
        ciborium::into_writer(self, &mut buf).expect("CBOR encode should not fail");
        buf
    }

    pub fn decode(data: &[u8]) -> Option<Self> {
        if data.is_empty() {
            return None;
        }
        ciborium::from_reader(data).ok()
    }
}

// ---------------------------------------------------------------------------
// ItemStack convenience constructors
// ---------------------------------------------------------------------------

impl ItemStack {
    /// Create a block_stack item for the given block type.
    pub fn block(namespace: u32, block_type: u32, count: u32) -> Self {
        let meta = BlockStackMeta {
            namespace,
            block_type,
            extra_data: Vec::new(),
        };
        Self {
            item: Item {
                namespace: ITEM_BLOCK_STACK.0,
                item_type: ITEM_BLOCK_STACK.1,
                data: meta.encode(),
            },
            count,
        }
    }

    /// If this is a block_stack item, decode the block data (with default orientation).
    pub fn to_block_data(&self) -> Option<BlockData> {
        if (self.item.namespace, self.item.item_type) != ITEM_BLOCK_STACK {
            return None;
        }
        let meta = BlockStackMeta::decode(&self.item.data)?;
        Some(BlockData {
            namespace: meta.namespace,
            block_type: meta.block_type,
            orientation: TesseractOrientation::IDENTITY,
            extra_data: meta.extra_data,
            scale_exp: 0,
        })
    }

    /// If this is a block_stack item, return the (namespace, block_type).
    pub fn block_type_key(&self) -> Option<(u32, u32)> {
        if (self.item.namespace, self.item.item_type) != ITEM_BLOCK_STACK {
            return None;
        }
        let meta = BlockStackMeta::decode(&self.item.data)?;
        Some((meta.namespace, meta.block_type))
    }
}
